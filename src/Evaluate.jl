#------------------------------------------------------------------------------------------------
# Functions for evaluating the model 
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
# for SMArtCARE data 
#------------------------------------------------------------------------------------------------

# measure prediction performance at each time point after the first, for both regression and our approach. 
function eval_prediction(m::odevae, testdata::SMATestData, n_future_tps::Int=1)
    ODEprederrs = []
    OLSprederrs = []
    interceptprederrs = []
    locfprederrs = []
    for ind in 1:length(testdata.xs)
        # get data
        curxs, curxs_baseline, curtvals = testdata.xs[ind], testdata.xs_baseline[ind], testdata.tvals[ind]
        # get latent representation 
        latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        # get ODe params
        params = vec(m.ODEnet(curxs_baseline))
        ODEparams = m.dynamics(params)
        # init prederrs 
        curODEprederr = zeros(eltype(latentμ), size(latentμ,1))
        curOLSprederr = zeros(eltype(latentμ), size(latentμ,1))
        curinterceptprederr = zeros(eltype(latentμ), size(latentμ,1))
        curlocfprederr = zeros(eltype(latentμ), size(latentμ,1))

        for tp_ind in 1:length(curtvals)-1
            for future_tp_ind in 1:n_future_tps
                curODEpred = generalsolution(curtvals[tp_ind+future_tp_ind]-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1]
                curODEprederr += (latentμ[:,tp_ind+future_tp_ind] .- curODEpred).^2
            end
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(curtvals[tp_ind:tp_ind+n_future_tps]))
            curOLSpred = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t),
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSpred[1,:] .- latentμ[:,tp_ind]
            curOLSpred = mapslices(x -> x - offset, curOLSpred, dims=2)

            curOLSprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- curOLSpred[2:end,:]').^2, dims=2)
            # intercept 
            interceptpred = mean(latentμ[:,1:tp_ind], dims=2)
            curinterceptprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- interceptpred).^2, dims=2)
            # LOCF 
            curlocfprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- latentμ[:,tp_ind]).^2, dims=2)

        end
        push!(ODEprederrs, curODEprederr)
        push!(OLSprederrs, vec(curOLSprederr))
        push!(interceptprederrs, vec(curinterceptprederr))
        push!(locfprederrs, vec(curlocfprederr))

    end
    return ODEprederrs, OLSprederrs, locfprederrs, interceptprederrs
end

function eval_reconstructed_prediction(m::odevae, testdata::SMATestData, n_future_tps::Int=1)
    ODEprederrs = []
    OLSprederrs = []
    interceptprederrs = []
    locfprederrs = []

    for ind in 1:length(testdata.xs)
        # get data
        curxs, curxs_baseline, curtvals = testdata.xs[ind], testdata.xs_baseline[ind], testdata.tvals[ind]
        # get latent representation 
        latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        # get ODe params
        params = vec(m.ODEnet(curxs_baseline))
        ODEparams = m.dynamics(params)
        # init prederrs 
        curODEprederr = zeros(eltype(curxs), size(curxs,1))
        curOLSprederr = zeros(eltype(curxs), size(curxs,1))
        curinterceptprederr = zeros(eltype(curxs), size(curxs,1))
        curlocfprederr = zeros(eltype(curxs), size(curxs,1))

        curxs_ODE = curxs_OLS = curxs_LOCF = curxs_IC = similar(curxs)
        for tp_ind in 1:length(curtvals)-1
            for future_tp_ind in 1:n_future_tps
                curODEpred = generalsolution(curtvals[tp_ind+future_tp_ind]-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1]
                reconstructedODEpred = m.decodedμ(m.decoder(curODEpred))
                curODEprederr += (curxs[:,tp_ind+future_tp_ind] .- reconstructedODEpred).^2
            end
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(curtvals[tp_ind:tp_ind+n_future_tps]))
            curOLSpred = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t),
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSpred[1,:] .- latentμ[:,tp_ind]
            curOLSpred = mapslices(x -> x - offset, curOLSpred, dims=2)
            reconstructedOLSpred = m.decodedμ(m.decoder(curOLSpred))

            curOLSprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- reconstructedOLSpred[:,2:end]).^2, dims=2)
            # intercept 
            interceptpred = mean(latentμ[:,1:tp_ind], dims=2)
            reconstructedinterceptpred = m.decodedμ(m.decoder(interceptpred))
            curinterceptprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- reconstructedinterceptpred).^2, dims=2)
            # LOCF 
            curlocfprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- m.decodedμ(m.decoder(latentμ[:,tp_ind]))).^2, dims=2)

        end
        push!(ODEprederrs, curODEprederr)
        push!(OLSprederrs, vec(curOLSprederr))
        push!(interceptprederrs, vec(curinterceptprederr))
        push!(locfprederrs, vec(curlocfprederr))

    end
    return ODEprederrs, OLSprederrs, locfprederrs, interceptprederrs
end

function get_reconstructed_prediction(m::odevae, testdata::SMATestData)

    pred_xs_ODE = []
    pred_xs_OLS = []
    pred_xs_LOCF = []
    pred_xs_IC = []

    for ind in 1:length(testdata.xs)
        # get data
        curxs, curxs_baseline, curtvals = testdata.xs[ind], testdata.xs_baseline[ind], testdata.tvals[ind]
        # get latent representation 
        latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        # get ODE params
        params = vec(m.ODEnet(curxs_baseline))
        ODEparams = m.dynamics(params)

        # init reconstructed arrays  
        curxs_ODE = zero(curxs)
        curxs_OLS = zero(curxs)
        curxs_LOCF = zero(curxs)
        curxs_IC = zero(curxs)

        curxs_ODE[:,1] = m.decodedμ(m.decoder(latentμ[:,1]))
        curxs_OLS[:,1] = m.decodedμ(m.decoder(latentμ[:,1]))
        curxs_LOCF[:,1] = m.decodedμ(m.decoder(latentμ[:,1]))
        curxs_IC[:,1] = m.decodedμ(m.decoder(latentμ[:,1]))

        for tp_ind in 1:length(curtvals)-1
            # ODE 
            curODEpred = generalsolution(curtvals[tp_ind+1]-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1]
            reconstructedODEpred = m.decodedμ(m.decoder(curODEpred))
            curxs_ODE[:,tp_ind+1] = reconstructedODEpred

            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(curtvals[tp_ind:tp_ind+1]))
            curOLSpred = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t),
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSpred[1,:] .- latentμ[:,tp_ind]
            curOLSpred = mapslices(x -> x - offset, curOLSpred, dims=2)
            reconstructedOLSpred = m.decodedμ(m.decoder(curOLSpred))
            curxs_OLS[:,tp_ind+1] = reconstructedOLSpred[:,2]

            # intercept 
            interceptpred = mean(latentμ[:,1:tp_ind], dims=2)
            reconstructedinterceptpred = m.decodedμ(m.decoder(interceptpred))
            curxs_IC[:,tp_ind+1] = reconstructedinterceptpred

            # LOCF 
            curxs_LOCF[:,tp_ind+1] = m.decodedμ(m.decoder(latentμ[:,tp_ind]))
        end
        push!(pred_xs_ODE, curxs_ODE)
        push!(pred_xs_OLS, curxs_OLS)
        push!(pred_xs_LOCF, curxs_LOCF)
        push!(pred_xs_IC, curxs_IC)

    end
    return pred_xs_ODE, pred_xs_OLS, pred_xs_LOCF, pred_xs_IC
end

function evaluate_reconstruction_items(orig_xs, rec_xs)
    # assuming transformation has been applied to both -- re-transform both and calculate MSE
    prederrs = fill(-1.0f0, length(orig_xs))
    for ind in 1:length(orig_xs)
        orig_items = round.(transform_back(orig_xs[ind]))
        rec_items = transform_back(rec_xs[ind])
        prederrs[ind] = mean((orig_items .- rec_items).^2)
    end
    return prederrs
end

transform_back(X) = (sigmoid.(X) .- 0.1f0) .* 2.5f0

#------------------------------------------------------------------------------------------------
# for simulated data 
#------------------------------------------------------------------------------------------------

function eval_prediction(m::odevae, data::simdata, n_future_tps::Int=1)
    ODEprederrs = []
    OLSprederrs = []
    interceptprederrs = []
    locfprederrs = []
    for ind in 1:length(data.xs)
        # get data
        curxs, curxs_baseline, curtvals = data.xs[ind], data.x_baseline[ind], data.tvals[ind]
        # get latent representation 
        latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        # get ODe params
        params = vec(m.ODEnet(curxs_baseline))
        ODEparams = m.dynamics(params)
        # init prederrs 
        curODEprederr = zeros(eltype(latentμ), size(latentμ,1))
        curOLSprederr = zeros(eltype(latentμ), size(latentμ,1))
        curinterceptprederr = zeros(eltype(latentμ), size(latentμ,1))
        curlocfprederr = zeros(eltype(latentμ), size(latentμ,1))

        for tp_ind in 1:length(curtvals)-1
            for future_tp_ind in 1:n_future_tps
                curODEpred = generalsolution(curtvals[tp_ind+future_tp_ind]-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1]
                curODEprederr += (latentμ[:,tp_ind+future_tp_ind] .- curODEpred).^2
            end
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(curtvals[tp_ind:tp_ind+n_future_tps]))
            curOLSpred = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t),
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSpred[1,:] .- latentμ[:,tp_ind]
            curOLSpred = mapslices(x -> x - offset, curOLSpred, dims=2)

            curOLSprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- curOLSpred[2:end,:]').^2, dims=2)
            # intercept 
            interceptpred = mean(latentμ[:,1:tp_ind], dims=2)
            curinterceptprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- interceptpred).^2, dims=2)
            # LOCF 
            curlocfprederr += sum((latentμ[:,tp_ind+1:tp_ind+n_future_tps] .- latentμ[:,tp_ind]).^2, dims=2)

        end
        push!(ODEprederrs, curODEprederr)
        push!(OLSprederrs, vec(curOLSprederr))
        push!(interceptprederrs, vec(curinterceptprederr))
        push!(locfprederrs, vec(curlocfprederr))

    end
    return ODEprederrs, OLSprederrs, locfprederrs, interceptprederrs
end

function eval_reconstructed_prediction(m::odevae, data::simdata, n_future_tps::Int=1)
    ODEprederrs = []
    OLSprederrs = []
    interceptprederrs = []
    locfprederrs = []

    for ind in 1:length(data.xs)
        # get data
        curxs, curxs_baseline, curtvals = data.xs[ind], data.x_baseline[ind], data.tvals[ind]
        # get latent representation 
        latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
        # get ODe params
        params = vec(m.ODEnet(curxs_baseline))
        ODEparams = m.dynamics(params)
        # init prederrs 
        curODEprederr = zeros(eltype(curxs), size(curxs,1))
        curOLSprederr = zeros(eltype(curxs), size(curxs,1))
        curinterceptprederr = zeros(eltype(curxs), size(curxs,1))
        curlocfprederr = zeros(eltype(curxs), size(curxs,1))

        curxs_ODE = curxs_OLS = curxs_LOCF = curxs_IC = similar(curxs)
        for tp_ind in 1:length(curtvals)-1
            for future_tp_ind in 1:n_future_tps
                curODEpred = generalsolution(curtvals[tp_ind+future_tp_ind]-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1]
                reconstructedODEpred = m.decodedμ(m.decoder(curODEpred))
                curODEprederr += (curxs[:,tp_ind+future_tp_ind] .- reconstructedODEpred).^2
            end
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(curtvals[tp_ind:tp_ind+n_future_tps]))
            curOLSpred = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t),
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSpred[1,:] .- latentμ[:,tp_ind]
            curOLSpred = mapslices(x -> x - offset, curOLSpred, dims=2)
            reconstructedOLSpred = m.decodedμ(m.decoder(curOLSpred))

            curOLSprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- reconstructedOLSpred[:,2:end]).^2, dims=2)
            # intercept 
            interceptpred = mean(latentμ[:,1:tp_ind], dims=2)
            reconstructedinterceptpred = m.decodedμ(m.decoder(interceptpred))
            curinterceptprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- reconstructedinterceptpred).^2, dims=2)
            # LOCF 
            curlocfprederr += sum((curxs[:,tp_ind+1:tp_ind+n_future_tps] .- m.decodedμ(m.decoder(latentμ[:,tp_ind]))).^2, dims=2)

        end
        push!(ODEprederrs, curODEprederr)
        push!(OLSprederrs, vec(curOLSprederr))
        push!(interceptprederrs, vec(curinterceptprederr))
        push!(locfprederrs, vec(curlocfprederr))

    end
    return ODEprederrs, OLSprederrs, locfprederrs, interceptprederrs
end