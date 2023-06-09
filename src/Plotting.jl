using GLM

mutable struct SMATestData
    test::String
    xs::Vector{Matrix{Float32}}
    xs_baseline::Vector{Vector{Float32}}
    tvals::Vector{Vector{Float32}}
    ids::Vector
end

function createindividualplot(m::odevae, testdata::SMATestData, args::LossArgs, patient_id; 
    axislabs::Bool=false, title::String="", showOLS::Bool=true)
    idx=findall(x -> x == patient_id, testdata.ids)

    if length(idx) > 1
        error("patient ID $patient_id not unique!")
    else
        idx = idx[1]
    end
    curxs, curxs_baseline, curtvals = testdata.xs[idx], testdata.xs_baseline[idx], testdata.tvals[idx]
    latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    params = vec(m.ODEnet(curxs_baseline))
    ODEparams = m.dynamics(params)
    trange = Float32.(minimum(curtvals):0.1:maximum(curtvals)+1)
    if args.firstonly
        smoothμs = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in trange]...)
    else    
        solarray = [generalsolution(solveatt - curtvals[startind], latentμ[:,startind], ODEparams...)[1] for startind in 1:length(curtvals), solveatt in trange]
        #solarray = [get_solution(startind, targetind, curtvals, latentμ, ODEparams) for startind in 1:length(curtvals), targetind in 1:length(curtvals)]
        smoothμs = hcat([get_smoothμ(targetind, curtvals, solarray, false, false) for targetind in 1:length(trange)]...)
        #smoothμs = hcat([get_smoothμ(solveatt, curtvals, latentμ, latentlogσ, ODEparams, args.weighting, false) for solveatt in trange]...)
    end
    curplot = plot(collect(trange), smoothμs', line=(2, ["#1f77b4" "#ff7f0e"]), labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"])
    if showOLS
        OLSfit = hcat(predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:])))), predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:])))))
        plot!(curtvals, OLSfit, line = (2,"red"), label ="")
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title="$patient_id")
    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end

function createindividualplot_piecewise(m::odevae, testdata::SMATestData, patient_id; 
    title::String="", showOLS::Bool=true, axislabs::Bool=false)

    idx=findall(x -> x == patient_id, testdata.ids)
    idx = idx[1]
    curxs, curxs_baseline, curtvals = testdata.xs[idx], testdata.xs_baseline[idx], testdata.tvals[idx]
    latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    params = vec(m.ODEnet(curxs_baseline))
    ODEparams = m.dynamics(params)
    trange = minimum(curtvals):0.1:maximum(curtvals)+0.1
    curplot = plot()
    for tp_ind in 1:length(curtvals)-1
        curtrange = curtvals[tp_ind]:0.1:curtvals[tp_ind+1]
        cursmoothμ = hcat([generalsolution(curt-curtvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1] for curt in curtrange]...)
        labels = (tp_ind == length(curtvals)-1) ? [L"\mathrm{local~ODE~solution~}\widetilde{\mu_1}" L"\mathrm{local~ODE~solution~}\widetilde{\mu_2}"] : ""
        plot!(collect(curtrange), cursmoothμ', 
                    line=(2, ["#1f77b4" "#ff7f0e"]), 
                    labels = labels, 
                    ribbon=sqrt.(exp.(latentlogσ[:,tp_ind]')), 
                    fillcolor = ["#c6dbef" "#fdd0a2"])
        #push!(smoothμs, cursmoothμ)
        if showOLS
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(curtvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(collect(curtrange)))
            curOLSfit = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t), 
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSfit[1,:] .- latentμ[:,tp_ind]
            curOLSfit = mapslices(x -> x - offset, curOLSfit, dims=2)
            #push!(OLSfits, curOLSfit)
            label = (tp_ind == length(curtvals)-1) ? L"\mathrm{linear~regression}" : ""    
            plot!(collect(curtrange), curOLSfit, line=(2, "#7b4173"), label = label)
            #plot!(curtvals[tp_ind:tp_ind+1], vcat(latentμ[:, tp_ind]', curOLSfit), line=(2, "red"), label = label)
        end
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 4, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 4, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title=title)

    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    
    return curplot
end

function plot_selected_ids_piecewise(m::odevae, testdata::SMATestData, selected_ids::Array; showOLS::Bool=true)
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        push!(sel_array, createindividualplot_piecewise(m, testdata, patient_id, title="$(patient_id)", showOLS=showOLS))
    end
    panelplot = plot(sel_array..., layout=(Int(length(selected_ids)/4),4), legend=false, size=(1200,round(200/3)*length(selected_ids)))
    return panelplot
end

function plot_selected_ids(m::odevae, testdata::SMATestData, args::LossArgs, selected_ids::Array; showOLS::Bool=true)
    sel_array = []
    for (ind, patient_id) in enumerate(selected_ids)
        push!(sel_array, createindividualplot(m, testdata, args, patient_id, title="$(patient_id)", showOLS=showOLS))
    end
    panelplot = plot(sel_array..., layout=(Int(length(selected_ids)/4),4), legend=false, size=(1200,round(200/3)*length(selected_ids)))
    return panelplot
end


#------------------------------
# Simulated data
#------------------------------

function plot_truesolution(group, data::simdata, t_range, sol_group1, sol_group2; showdata=true)
    if group == 1
        sol = sol_group1
        groupinds = data.group1
        legendposition = :topleft
    else
        sol = sol_group2
        groupinds = data.group2
        legendposition = :topright
    end
    curplot = plot(t_range, sol',
                label = [L"\mathrm{true~solution~}z_1" L"\mathrm{true~solution~}z_2"],
                legend = legendposition,
                legendfontsize = 12,
                line=(3, ["#ff7f0e" "#1f77b4"])
                )
    if !showdata
        return curplot
    else
        for ind in 1:length(data.xs[groupinds])
            for var in 1:size(data.xs[groupinds][1],1)
                color = "#ffbb78" 
                if var > 5
                    color = "#aec7e8"
                end
                Plots.scatter!(data.tvals[groupinds][ind], data.xs[groupinds][ind][var,:], label="", marker=(:c,6,color))
            end
        end
    end
    return curplot
end

function createindividualplot(m::odevae, data::simdata, idx::Int, sol::Matrix, trange, args::LossArgs; 
    title::String="", 
    showtruesol::Bool=true,
    axislabs::Bool=false, 
    showOLS::Bool=true, 
    colors_truesol::Array{String} = ["#ff7f0e" "#1f77b4"]
    )
    #        
    curxs, curxs_baseline, curtvals = data.xs[idx], data.x_baseline[idx], data.tvals[idx]
    latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    params = vec(m.ODEnet(curxs_baseline))
    ODEparams = m.dynamics(params)
    if showtruesol
        curplot = plot(trange, sol',
            label = [L"\mathrm{true~solution~}u_1" L"\mathrm{true~solution~}u_2"],
            legend = :topleft,
            legendfontsize = 12,
            line=([:dot :dot], 3, colors_truesol)
        )
    else
        curplot = plot(xlimits = (0, 10))
    end

    if args.firstonly
        smoothμs = hcat([generalsolution(tp, latentμ[:,1], ODEparams...)[1] for tp in trange]...)
    else
        solarray = [generalsolution(solveatt - curtvals[startind], latentμ[:,startind], ODEparams...)[1] for startind in 1:length(curtvals), solveatt in trange]
        smoothμs = hcat([get_smoothμ(targetind, curtvals, solarray, false, false) for targetind in 1:length(trange)]...)
    end
    plot!(trange, smoothμs', line=(2, ["#1f77b4" "#ff7f0e"]), labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"])
    if showOLS
        OLSfit = hcat(predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:])))), predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:])))))
        plot!(curtvals, OLSfit, line = (2,"red"), label ="")
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}")
    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end

function eval_z_trajectories(m::odevae, data::simdata, inds::Array{Int}, 
    sol_group1::Matrix, sol_group2::Matrix, t_range, args::LossArgs; 
    title::String="", 
    showtruesol::Bool=true,
    axislabs::Bool=false, 
    showOLS::Bool=true, 
    swapcolorcoding::Bool=false
    )
    plotarray=[]
    for ind in sort(inds)
        colors_truesol = swapcolorcoding ? ["#1f77b4" "#ff7f0e"] : ["#ff7f0e" "#1f77b4"]
        sol = ind ∈ data.group1 ? sol_group1 : sol_group2
        curplot = createindividualplot(m, data, ind, sol, t_range, args; 
            title="$ind", 
            showtruesol = showtruesol,
            axislabs=axislabs, 
            showOLS=showOLS, 
            colors_truesol=colors_truesol
        )
        #for tp in 1:length(tvals[ind])
        #    Plots.scatter!(repeat([tvals[ind][tp]], length(curxs[:,1])), origxsotherts[:,tp]; marker=(:c, 3, "#bab0ac"), alpha=0.5)
        #end
        push!(plotarray, curplot)
    end
    myplot = plot(plotarray[:]..., layout=(2,3), plot_title=title, legend=false)#layout=(3,3))
    display(myplot)
end

function createindividualplot_piecewise(m::odevae, data::simdata, ind::Int, sol::Matrix, t_range; 
    title::String="", 
    axislabs::Bool=false, 
    showtruesol::Bool=true,
    showOLS::Bool=true, 
    showglobalOLS::Bool=false,
    colors_truesol::Array{String} = ["#ff7f0e" "#1f77b4"]
    )
    curxs, curxs_baseline, curtvals = data.xs[ind], data.x_baseline[ind], copy(data.tvals[ind])
    latentμ, latentlogσ = m.encodedμ(m.encoder(curxs)), m.encodedlogσ(m.encoder(curxs))
    params = vec(m.ODEnet(curxs_baseline))
    ODEparams = m.dynamics(params)
    if showtruesol
        curplot = plot(t_range, sol',
        label = [L"\mathrm{true~solution~}z_1" L"\mathrm{true~solution~}z_2"],
        legend = :topleft,
        legendfontsize = 12,
        line=([:dot :dot], 3, colors_truesol)
    )
    else
        curplot = plot(xlimits = (0, 10))
    end
    mod_tvals = (curtvals[end] == 10.0f0) ? copy(curtvals) : [curtvals..., 10.0f0]
    for tp_ind in 1:length(mod_tvals)-1
        curtrange = mod_tvals[tp_ind]:0.1:mod_tvals[tp_ind+1]
        cursmoothμ = hcat([generalsolution(curt-mod_tvals[tp_ind], latentμ[:,tp_ind], ODEparams...)[1] for curt in curtrange]...)
        labels = (tp_ind == length(mod_tvals)-1) ? [L"\mathrm{local~ODE~solution~}\widetilde{\mu_1}" L"\mathrm{local~ODE~solution~}\widetilde{\mu_2}"] : ""
        plot!(collect(curtrange), cursmoothμ', 
                    line=(2, ["#1f77b4" "#ff7f0e"]), 
                    labels = labels, 
                    ribbon=sqrt.(exp.(latentlogσ[:,tp_ind]')), 
                    fillcolor = ["#c6dbef" "#fdd0a2"])
        #push!(smoothμs, cursmoothμ)
        if showOLS
            # OLS 
            curOLSdf_1 = DataFrame(X=Float64.(mod_tvals[1:tp_ind]), Y=Float64.(latentμ[1,1:tp_ind]))
            curOLSdf_2 = DataFrame(X=Float64.(mod_tvals[1:tp_ind]), Y=Float64.(latentμ[2,1:tp_ind]))
            pred_t = DataFrame(X=Float64.(collect(curtrange)))
            curOLSfit = hcat(predict(lm(@formula(Y~X), curOLSdf_1), pred_t), 
                            predict(lm(@formula(Y~X), curOLSdf_2), pred_t)
            )
            offset = curOLSfit[1,:] .- latentμ[:,tp_ind]
            curOLSfit = mapslices(x -> x - offset, curOLSfit, dims=2)
            #push!(OLSfits, curOLSfit)
            label = (tp_ind == length(mod_tvals)-1) ? L"\mathrm{linear~regression}" : ""    
            plot!(collect(curtrange), curOLSfit, line=(2, "red"), label = label)
            #plot!(curtvals[tp_ind:tp_ind+1], vcat(latentμ[:, tp_ind]', curOLSfit), line=(2, "red"), label = label)
        end
    end
    if showglobalOLS
        globalOLSfit = hcat(
            predict(
                lm(@formula(Y~X), 
                    DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:]))
                ),
                DataFrame(X=Float64.(mod_tvals))
            ), 
            predict(
                lm(@formula(Y~X), 
                    DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:]))
                ),
                DataFrame(X=Float64.(mod_tvals))
            )
        )
        plot!(mod_tvals, globalOLSfit, line = (2,"red"), label = L"\mathrm{linear~regression}")
    end

    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 4, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 4, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title=title)

    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end

    return curplot
end

function plot_selected_ids_piecewise(m::odevae, data::simdata, inds::Array{Int}, 
    sol_group1::Matrix, sol_group2::Matrix, t_range; 
    axislabs::Bool=false, 
    showtruesol::Bool=true,
    showtitle::Bool=true, 
    showOLS::Bool=true, 
    showglobalOLS::Bool=false,
    swapcolorcoding::Bool=false, 
    sort_inds::Bool=true
    )
    plotarray = []
    ind_iterator = sort_inds ? sort(inds) : inds
    for ind in ind_iterator
        title = showtitle ? "$ind" : ""
        colors_truesol = swapcolorcoding ? ["#1f77b4" "#ff7f0e"] : ["#ff7f0e" "#1f77b4"]
        sol = ind ∈ data.group1 ? sol_group1 : sol_group2
        curplot = createindividualplot_piecewise(m, data, ind, sol, t_range; 
            title=title, 
            showtruesol=showtruesol,
            axislabs=axislabs, 
            showOLS=showOLS, 
            showglobalOLS=showglobalOLS, 
            colors_truesol=colors_truesol
        )
        push!(plotarray, curplot)
    end
    panelplot = plot(plotarray..., layout=(Int(length(inds)/4),4), legend=false, size=(1200,round(200/3)*length(inds)))
    return panelplot
end