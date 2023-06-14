#------------------------------
# functions to define and manipulate ODE-VAE model 
#------------------------------

#------------------------------
# ODE solutions
#------------------------------

# calculate analytical solution from A, c, x0

"""
    generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32}) 

Calculates the analytical solution of a linear system of ODEs with constant coefficients 
    at a time point `t`, for a system matrix `A`, a constant vector `c` and an initial value `x0`.
    
Returns the solution `x(t)` and the matrix exponential `e^{At}`.
"""
function generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32})
    eAt = exp(A.*t)
    return eAt*(c + x0) - c, eAt
end

"""
    generalsolution(t, x0::Vector{Float32}, c::Vector{Float32}) 

Calculates the analytical solution of a constant system of ODEs at a time point `t`, 
    for a constant vector `c` and an initial value `x0`.
    
Returns the solution `x(t)` and the matrix exponential `e^{At}` = 1.0.
"""
function generalsolution(t, x0::Vector{Float32}, p::Vector{Float32}) # for drift only solution 
    return p.*t + x0, 1.0f0
end

# get parameters for each system 

"""
    params_fullinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system 
    with 6 parameters specified by the input vector `p` = [a11, a12, a21, a22, c1, c2].
"""
function params_fullinhomogeneous(p::Vector{Float32})
    if length(p) != 6
        error("2D inhomogeneous linear system requires 6 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    c = inv(A)*p[5:6]
    return A, c
end

"""
    params_offdiagonalinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system where 
    the system matrix has only off-diagonal elements (i.e., diagonal entries are zero), 
    specified by the input vector `p` = [a12, a21, c1, c2].
"""
function params_offdiagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only off-diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([0.0f0 p[1] p[2] 0.0f0], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

"""
    params_diagonalinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system where
    the system matrix is diagonal, specified by the input vector `p` = [a11, a22, c1, c2].
"""
function params_diagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

"""
    params_driftonly(p::Vector{Float32})

Returns the constant vector `c` of a 2D inhomogeneous linear system where
    the system matrix is zero, specified by the input vector `p` = [c1, c2].
"""
function params_driftonly(p::Vector{Float32})
    if length(p) != 2
        error("drift only solution requires 2 parameters, but p is of length $(length(p))")
    end
    return [p]
end

"""
    params_fullhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D homogeneous linear system
    with 4 parameters specified by the input vector `p` = [a11, a12, a21, a22], and `c` = [0, 0].
"""
function params_fullhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D homogeneous linear system requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    return A, zeros(Float32,2)
end

"""
    params_offdiagonalhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D homogeneous linear system where
    the system matrix is diagonal, specified by the input vector `p` = [a11, a22], and `c` = [0, 0].
"""
function params_diagonalhomogeneous(p::Vector{Float32})
    if length(p) != 2
        error("2D homogeneous linear system without interactions requires 2 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    return A, zeros(Float32,2)
end

#------------------------------
# define and initialize model 
#------------------------------
"""
    odevae

Struct for an ODE-VAE model, with the following fields:
- `p`: number of VAE input dimensions, i.e., number of time-dependent variables
- `q`: number of input dimensions for the baseline neural net, i.e., number of baseline variables
- `zdim`: number of latent dimensions
- `ODEnet`: neural net to map baseline variables to individual-specific ODE parameters 
    (number of ODE parameters depends on the ODE system specified by the `dynamics` function)
- `encoder`: neural net to map input data to latent space
- `encodedμ`: neural net layer parameterizing the mean of the latent space
- `encodedlogσ`: neural net layer parameterizing the log variance of the latent space
- `decoder`: neural net to map latent variable to reconstructed input data
- `decodedμ`: neural net layer parameterizing the mean of the reconstructed input data
- `decodedlogσ`: neural net layer parameterizing the log variance of the reconstructed input data
- `dynamics`: one of `params_fullinhomogeneous`, `params_offdiagonalinhomogeneous`, 
    `params_diagonalinhomogeneous`, `params_driftonly`, `params_fullhomogeneous`, 
    `params_offdiagonalhomogeneous`, `params_diagonalhomogeneous`: function to map a parameter vector
    (=the output of the `ODEnet`) to the system matrix and constant vector of the ODE system
"""
mutable struct odevae
    p::Int
    q::Int
    zdim::Int
    ODEnet
    encoder
    encodedμ 
    encodedlogσ 
    decoder
    decodedμ 
    decodedlogσ 
    dynamics::Function # either ODEprob or params for analytical solution function 
end

"""
    ModelArgs

Struct to store model arguments, can be constructed with keyword arguments to set the following fields:
- `p`: number of VAE input dimensions, i.e., number of time-dependent variables
- `q`: number of input dimensions for the baseline neural net, i.e., number of baseline variables
- `zdim`: number of latent dimensions
- `dynamics`: one of `params_fullinhomogeneous`, `params_offdiagonalinhomogeneous`, 
    `params_diagonalinhomogeneous`, `params_driftonly`, `params_fullhomogeneous`, 
    `params_offdiagonalhomogeneous`, `params_diagonalhomogeneous`: function to map a parameter vector
    (=the output of the `ODEnet`) to the system matrix and constant vector of the ODE system
- `seed`: random seed for reproducibility
- `bottleneck`: whether to use a bottleneck layer in the `ODEnet` 
    to reduce the number of effective parameters for higher-dimensional systems
- `init_scaled`: whether to initialize the `ODEnet` with scaled weights
- `scale_sigmoid`: scaling factor for the sigmoid function used to shift the ODE parameters 
    to a sensible range, acting as a prior
- `add_diagonal`: whether to add a diagonal transformation to output of the `ODEnet` to add
    flexibility after the sigmoid transformation
"""
@with_kw struct ModelArgs
    p::Int
    q::Int
    zdim::Int=2
    dynamics::Function
    seed::Int=1234
    bottleneck::Bool=false
    init_scaled::Bool=false
    scale_sigmoid::Real=1
    add_diagonal::Bool=true
end

"""
    LossArgs

Struct to store loss arguments, can be constructed with keyword arguments to set the following fields:
- `λ_μpenalty`: weight for the penalty that encourages consistency of the mean before and after solving the ODEs
- `λ_variancepenalty`: weight for the penalty on the variance of the ODE estimator
- `variancepenaltytype`: one of `:ratio_sum`, `:sum_ratio`, `:log_diff`: 
    type of penalty on the variance of the ODE estimator
- `variancepenaltyoffset`: offset used in the penalty on the variance of the latent space
- `firstonly`: whether to use only the first time point for solving the ODE (if `false`, 
    an ODE is solved with each time point as initial condition and the individual solutions are averaged)
- `weighting`: whether to calculate inverse-variance weights for the contribution of other time points 
    in the ODE trajectory estimator or use just equal weights for all ODE solutions
- `skipt0`: whether to skip the first time point in the ODE estimator 
    (to prevent the model from using just the initial condition and pushing the weights of all other solutions to zero)
"""
@with_kw struct LossArgs
    λ_μpenalty::Float32 = 0.0f0
    λ_variancepenalty::Float32 = 0.0f0
    variancepenaltytype::Symbol = :ratio_sum # :sum_ratio, log_diff
    variancepenaltyoffset::Float32 = 1.0f0
    firstonly::Bool=false
    weighting::Bool=false
    skipt0::Bool=false
end

downscaled_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(0.01f0/sum(dims)) # smaller weights initialisation

function get_nODEparams(dynamics::Function)
    if dynamics ∈ [params_driftonly, params_diagonalhomogeneous]
        nODEparams = 2
    elseif dynamics ∈[params_fullhomogeneous, params_offdiagonalinhomogeneous, params_diagonalinhomogeneous]
        nODEparams = 4 
    elseif dynamics == params_fullinhomogeneous
        nODEparams = 6 
    else
        error("unsupported model dynamics")
    end
    return nODEparams
end

# initialise model
"""
    odevae(modelargs::ModelArgs)

Function to initialize the ODE-VAE model according to the arguments passed in `modelargs`.

Returns an `odevae` model.
"""
function odevae(modelargs::ModelArgs)
    nODEparams = get_nODEparams(modelargs.dynamics)
    myinit = modelargs.init_scaled ? downscaled_glorot_uniform : Flux.glorot_uniform 
    shift(arg) = (sigmoid(arg).-0.5f0)/modelargs.scale_sigmoid
    # seed
    Random.seed!(modelargs.seed)
    # parameter network
    if !modelargs.bottleneck
        ODEnet = [Dense(modelargs.q, modelargs.q, tanh, init=myinit),
                        Dense(modelargs.q, nODEparams, arg ->(shift(arg)), init=myinit)
        ]
    else
        ODEnet = [Dense(modelargs.q, nODEparams, tanh, init=myinit),
                    Dense(nODEparams, 2, tanh, init=myinit),
                    Dense(2, nODEparams, arg ->(shift(arg)), init=myinit)
        ]
    end
    if modelargs.add_diagonal
        ODEnet = Chain(ODEnet..., Flux.Diagonal(nODEparams))
    else
        ODEnet = Chain(ODEnet...)
    end
    #   VAE encoder
    Dz, Dh = modelargs.zdim, modelargs.p
    encoder, encodedμ, encodedlogσ = Dense(modelargs.p, Dh, arg ->(tanh.(arg) .+ 1)), Dense(Dh, Dz), Chain(Dense(Dh, Dz, arg -> -Flux.relu(arg)), Flux.Diagonal(Dz))
    # VAE decoder
    decoder, decodedμ, decodedlogσ = Dense(Dz, Dh, tanh), Dense(Dh, modelargs.p), Dense(Dh, modelargs.p)

    model = odevae(modelargs.p, modelargs.q, modelargs.zdim, ODEnet, encoder, encodedμ, encodedlogσ, decoder, decodedμ, decodedlogσ, modelargs.dynamics)
    return model
end

#------------------------------
# define penalties
#------------------------------

function μ_penalty(datatvals, latentμ, ODEparams)
    penalty = 0.0f0
    for i in 1:size(latentμ,2) 
        for (tind, solveatt) in enumerate(datatvals) # make pred with current x0 for every tval
            pred, varfactor = generalsolution(solveatt - datatvals[i], latentμ[:,i], ODEparams...)
            if solveatt != datatvals[i]
                penalty += sqrt(sum((pred .- latentμ[:,tind]).^2)) # squared difference between prediction and actual datapoint 
            end
        end
    end
    return penalty/(length(datatvals)^2)
end

#------------------------------
# define model functions 
#------------------------------

latentz(μ, logσ) = μ .+ sqrt.(exp.(logσ)) .* randn(Float32,size(μ)...) # sample latent z,

kl_q_p(μ, logσ) = 0.5f0 .* sum(exp.(logσ) + μ.^2 .- 1.0f0 .- (logσ),dims=1)

#logp_x_z(m::odevae, x, z) = sum(logpdf.(Normal.(m.decodedμ(m.decoder(z)), sqrt.(exp.(m.decodedlogσ(m.decoder(z))))), x),dims=1) # get reconstruction error

function logp_x_z(m::odevae, x::AbstractVecOrMat{S}, z::AbstractVecOrMat{S}) where S <: Real 
    μ = m.decodedμ(m.decoder(z))
    logσ = m.decodedlogσ(m.decoder(z))
    res = @fastmath (-(x .- μ).^2 ./ (2.0f0 .* exp.(logσ))) .- 0.5f0 .* (log(S(2π)) .+ logσ)
    return sum(res, dims=1)
end

sqnorm(x) = sum(abs2, x)
reg(m::odevae) = sum(sqnorm, Flux.params(m.decoder,m.decodedμ,m.decodedlogσ)) # regularisation term in loss

getparams(m::odevae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.ODEnet) # get parameters of VAE model

function getparams(m1::odevae, m2::odevae) 
    Flux.params(m1.encoder, m1.encodedμ, m1.encodedlogσ, 
                m1.decoder, m1.decodedμ, m1.decodedlogσ, 
                m2.encoder, m2.encodedμ, m2.encodedlogσ,
                m2.decoder, m2.decodedμ, m2.decodedlogσ,
                m1.ODEnet
    ) # get parameters of VAE model
end

function get_reconstruction(m::odevae, X, Y, t, args::LossArgs; sample::Bool=false)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    params = vec(m.ODEnet(Y))
    ODEparams = m.dynamics(params)
    if args.firstonly
        smoothμ = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in t]...)
    else
        @assert length(t) == size(latentμ,2)
        solarray = [get_solution(startind, targetind, t, latentμ, ODEparams) for startind in 1:length(t), targetind in 1:length(t)]
        smoothμ = hcat([get_smoothμ(targetind, t, solarray, args.weighting, args.skipt0) for targetind in 1:length(t)]...)
    end
    if sample
        z = latentz.(smoothμ, latentlogσ)
    else
        z = smoothμ
    end
    decodedμ = m.decodedμ(m.decoder(z))
    reconstructed_X = decodedμ
    if sample
        decodedlogσ = m.decodedlogσ(m.decoder(z))
        reconstructed_X = rand.(Normal.(decodedμ, sqrt.(exp.(decodedlogσ))))
    end
    return reconstructed_X
end

function get_solution(startind, targetind, tvals, latentμ, ODEparams)
    solveatt = tvals[targetind]
    tstart = tvals[startind]
    return generalsolution(solveatt - tstart, latentμ[:,startind], ODEparams...)[1]
end

function get_smoothμ(targetind::Int, tvals::Vector{Float32}, solarray, weighting::Bool, skipt0::Bool)
    weightedpred = sum_weights = zero(solarray[1,1])
    for startind in 1:length(tvals)
        if skipt0 && (startind == targetind) && (length(tvals) > 1)
            continue
        end    
        pred = solarray[startind,targetind]
        if weighting
            # for every starting point between solveatt and datatvals[i], make a prediction for solveatt, 
            # take the empirical variance of the solutions
            var_range = startind < targetind ? (startind:targetind) : (targetind:startind)
            weight = 1.0f0 ./ var(solarray[var_range, targetind])
        else
            weight = one.(pred)
        end
        weightedpred += pred.*weight
        sum_weights += weight
    end
    return weightedpred ./ sum_weights
end

function get_smoothμs(tvals, latentμ, ODEparams, weighting::Bool, skipt0::Bool)
    @assert length(tvals) == size(latentμ,2)
    solarray = [get_solution(startind, targetind, tvals, latentμ, ODEparams) for startind in 1:length(tvals), targetind in 1:length(tvals)]
    smoothμs = hcat([get_smoothμ(targetind, tvals, solarray, weighting, skipt0) for targetind in 1:length(tvals)]...)
    return smoothμs
end

#------------------------------
# loss functions 
#------------------------------

"""
    loss(X, Y, t, m::odevae; args::LossArgs)

Compute the loss of the ODE-VAE model `m` on a batch of data, consisting of 
    time-dependent variables `X`, baseline variables `Y` and time point `t`. 

Details of the loss function behaviour, including additional penalties, are controlled by the 
    keyword arguments `args` of type `LossArgs`, see `?LossArgs` for details.

Returns the mean ELBO, where the ODE estimator of the underlying trajectory is used to decode the latent 
    value at the time points `t` and obtain a reconstruction according to these smooth latent dynamics 
    as specified by the ODE system.
"""
function loss(X, Y, t, m::odevae; args::LossArgs)
    latentμ, latentlogσ = m.encodedμ(m.encoder(X)), m.encodedlogσ(m.encoder(X))
    params = vec(m.ODEnet(Y))
    ODEparams = m.dynamics(params)
    # smoothμ = Array(solve(m.ODEprob, Tsit5(), u0 = latentμ[1,1], latentμ[2,1], p=curparams, saveat=dt))[:,curts]
    if args.firstonly
        smoothμ = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in t]...)
    else
        @assert length(t) == size(latentμ,2)
        solarray = [get_solution(startind, targetind, t, latentμ, ODEparams) for startind in 1:length(t), targetind in 1:length(t)]
        smoothμ = hcat([get_smoothμ(targetind, t, solarray, args.weighting, args.skipt0) for targetind in 1:length(t)]...)
    end
    z = latentz.(smoothμ, latentlogσ)
    ELBO = 1.0f0 .* logp_x_z(m, X, z) .- 0.5f0 .* kl_q_p(smoothμ, latentlogσ)# previous KL weight: 0.5
    penalties = 0.0f0
    if args.λ_μpenalty > 0.0f0
        penalties += args.λ_μpenalty * μ_penalty(t, latentμ, ODEparams)
    end
    if args.λ_variancepenalty > 0.0f0
        offset = args.variancepenaltyoffset
        if args.variancepenaltytype::Symbol == :ratio_sum
            var_ratio = (sum(mean(var(solarray[:,targetind]) for targetind in 1:length(t))) .+ offset) / (sum(var(latentμ, dims=2)) .+ offset)
        elseif args.variancepenaltytype::Symbol == :sum_ratio
            var_ratio = sum((mean(var(solarray[:,targetind]) for targetind in 1:length(t)) .+ offset) ./ (var(latentμ, dims=2) .+ offset))
        elseif args.variancepenaltytype::Symbol == :log_diff
            var_ratio = sum(mean(log.(var(solarray[:,targetind]) .+ offset) for targetind in 1:length(t)) .- log.(var(latentμ, dims=2) .+ offset))
        end
        penalties += args.λ_variancepenalty * var_ratio
    end
    lossval = mean(-ELBO) + 0.01f0*reg(m) + penalties # sum(-ELBO)
    return lossval
end

#------------------------------
# train model 
#------------------------------

"""
    train_model!(m::odevae, 
        xs, xs_baseline, tvals, 
        lr, epochs, args::LossArgs; 
        selected_ids=nothing, 
        verbose::Bool=true, 
        plotting::Bool=true
        )

Train the ODE-VAE model `m` on a dataset of time-dependent variables `xs`, 
    baseline variables `xs_baseline` and time points `tvals`. The structure of these 
    is assumed to be as in the `SMATestData` and `simdata` structs. 

# Arguments
- `m`: the ODE-VAE model to train
- `xs`: a vector of matrices of time-dependent variables for each patient
- `xs_baseline`: a vector of vectors of baseline variables for each patient
- `tvals`: a vector of vectors of time points for each patient
- `lr`: the learning rate of the ADAM optimizer
- `epochs`: the number of epochs to train for
- `args`: arguments controlling the loss function behaviour, see `?LossArgs` for details
- `selected_ids`: the IDs of the patients to plot during training to monitor progress,
    if `nothing` (default) then 12 random IDs are selected
- `verbose`: whether to print the epoch and loss value during training
- `plotting`: whether to visualize the learnt latent trajectories of selected patients 
    (those with the `selected_ids`)

# Returns 
- `m`: the trained ODE-VAE model
"""
function train_model!(m::odevae, 
    xs, xs_baseline, tvals, 
    lr, epochs, args::LossArgs; 
    selected_ids=nothing, 
    verbose::Bool=true, 
    plotting::Bool=true
    )
    if (isnothing(selected_ids) || length(selected_ids) != 12) && plotting
        selected_ids = rand(ids,12)
    end
    # prepare training
    ps = getparams(m)
    opt = ADAM(lr)
    trainingdata = zip(xs, xs_baseline, tvals);

    # callback 
    evalcb() = @show(mean(loss(data..., m, args=args) for data in trainingdata))

    # start training 
    state = copy(Random.default_rng());
    for epoch in 1:epochs
        verbose && @info epoch
        copy!(Random.default_rng(), state);
        for (X, Y, t) in trainingdata
            grads = Flux.gradient(ps) do 
                loss(X, Y, t, m, args=args)
            end
            Flux.Optimise.update!(opt, ps, grads)
        end
        state = copy(Random.default_rng());
        verbose && evalcb()
        plotting && display(plot_selected_ids(m, testdata, args, selected_ids))
        #evalcb_zs()
    end
    return m
end
