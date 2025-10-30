using GLM

"""
    createindividualplot(m::odevae, testdata::SMATestData, args::LossArgs, patient_id; 
        axislabs::Bool=false, title::String="", showOLS::Bool=true)

Plot the learned latent trajectory for a single SMArtCARE patient, comparing the ODE-smoothed
latent mean to encoder means (and optionally an OLS fit).

The smooth trajectory is obtained either from a single-IC rollout (`args.firstonly == true`)
or by averaging predictions from all time points as initial conditions.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `testdata::SMATestData`: dataset holding time series, baselines, ids, and time points
- `args::LossArgs`: controls the smoothing strategy (e.g., `firstonly`)
- `patient_id`: identifier present in `testdata.ids`
- `axislabs::Bool=false`: add axis labels (“time in months”, “value of latent representation”)
- `title::String=""`: subplot title
- `showOLS::Bool=true`: overlay a (global) piecewise-aligned linear regression in latent space

# Returns
- `curplot`: a `Plots.Plot` object with:
    - smooth ODE latent means over a dense time grid,
    - encoder latent means as scatter,
    - optional OLS line(s).
"""
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
    curplot = plot(collect(trange), smoothμs', 
        line=(3, ["#1f77b4" "#ff7f0e"]), 
        labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"]
    )
    if showOLS
        OLSfit = hcat(predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:])))), predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:])))))
        plot!(curtvals, OLSfit, line = (3, "#e70f4f", :dash), label ="")
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title="$patient_id")
    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end

"""
    createindividualplot_piecewise(m::odevae, testdata::SMATestData, patient_id; 
        title::String="", showOLS::Bool=true, axislabs::Bool=false)

Plot local, piecewise ODE solutions between consecutive observed time points for one SMArtCARE
patient, with uncertainty ribbons from the encoder’s log-variance and optional OLS overlays.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `testdata::SMATestData`: dataset holding time series, baselines, ids, and time points
- `patient_id`: identifier present in `testdata.ids`
- `title::String=""`: subplot title
- `showOLS::Bool=true`: overlay per-segment linear regressions aligned at each segment start
- `axislabs::Bool=false`: add axis labels

# Returns
- `curplot`: a `Plots.Plot` object with segment-wise ODE rolls, encoder means (scatter), and optional OLS fits.
"""
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
                    line=(3, ["#1f77b4" "#ff7f0e"]), 
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
            plot!(collect(curtrange), curOLSfit, line=(3, "#e70f4f", :dash), label = label) # 7b4173 # e45756 # d67195
            #plot!(curtvals[tp_ind:tp_ind+1], vcat(latentμ[:, tp_ind]', curOLSfit), line=(2, "red"), label = label)
        end
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title=title)

    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    
    return curplot
end

"""
    plot_selected_ids_piecewise(m::odevae, testdata::SMATestData, selected_ids::Array; 
        showOLS::Bool=true, layout=nothing, size=nothing, 
        save_plot::Bool=false, save_path::String="")

Create a panel of piecewise ODE plots for multiple SMArtCARE patients.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `testdata::SMATestData`: dataset with time series, baselines, ids, and time points
- `selected_ids::Array`: collection of patient IDs to plot (order preserved)
- `showOLS::Bool=true`: show per-segment OLS overlays in each subplot
- `layout=nothing`: layout tuple `(rows, cols)`; defaults to `(length(selected_ids)÷4, 4)`
- `size=nothing`: `(width, height)` in pixels; defaults to `(1200, round(200/3)*length(selected_ids))`
- `save_plot::Bool=false`: save figure to disk
- `save_path::String=""`: path passed to `savefig` if `save_plot == true`

# Returns
- `panelplot`: a `Plots.Plot` panel with the requested subplots.

# Notes
Saves the panel to `save_path` when `save_plot=true`.
"""
function plot_selected_ids_piecewise(m::odevae, testdata::SMATestData, selected_ids::Array; 
    showOLS::Bool=true, layout=nothing, size=nothing, save_plot::Bool=false, save_path::String="")

    sel_array = []

    for (ind, patient_id) in enumerate(selected_ids)
        push!(sel_array, createindividualplot_piecewise(m, testdata, patient_id, title="$(patient_id)", showOLS=showOLS))
    end

    if isnothing(layout)
        mylayout = (Int(length(selected_ids)/4),4)
    else
        mylayout = layout
    end

    # customize plotsize
    if isnothing(size)
        mysize = (1200,round(200/3)*length(selected_ids))
    else
        mysize = size
    end
    
    panelplot = plot(sel_array..., layout=mylayout, legend=false, size=mysize)

    save_plot && savefig(panelplot, save_path)

    return panelplot
end

"""
    plot_selected_ids(m::odevae, testdata::SMATestData, args::LossArgs, selected_ids::Array; 
        showOLS::Bool=true, layout=nothing, size=nothing, 
        save_plot::Bool=false, save_path::String="")

Create a panel of global ODE-smoothed latent trajectories for multiple SMArtCARE patients.
Uses the same smoothing strategy as `createindividualplot`.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `testdata::SMATestData`: dataset with time series, baselines, ids, and time points
- `args::LossArgs`: controls smoothing (e.g., `firstonly`)
- `selected_ids::Array`: collection of patient IDs to plot
- `showOLS::Bool=true`: show global OLS overlays
- `layout=nothing`: layout tuple `(rows, cols)`; defaults to `(length(selected_ids)÷4, 4)`
- `size=nothing`: `(width, height)` in pixels; defaults to `(1200, round(200/3)*length(selected_ids))`
- `save_plot::Bool=false`: save figure to disk
- `save_path::String=""`: path passed to `savefig` if `save_plot == true`

# Returns
- `panelplot`: a `Plots.Plot` panel with the requested subplots.

# Notes
Saves the panel to `save_path` when `save_plot=true`.
"""
function plot_selected_ids(m::odevae, testdata::SMATestData, args::LossArgs, selected_ids::Array; 
    showOLS::Bool=true, layout=nothing, size=nothing, save_plot::Bool=false, save_path::String="")

    sel_array = []

    for (ind, patient_id) in enumerate(selected_ids)
        push!(sel_array, createindividualplot(m, testdata, args, patient_id, title="$(patient_id)", showOLS=showOLS))
    end

    # customize layout
    if isnothing(layout)
        mylayout = (Int(length(selected_ids)/4),4)
    else
        mylayout = layout
    end

    # customize plotsize
    if isnothing(size)
        mysize = (1200,round(200/3)*length(selected_ids))
    else
        mysize = size
    end

    panelplot = plot(sel_array..., layout=mylayout, legend=false, size=mysize)

    save_plot && savefig(panelplot, save_path)

    return panelplot
end

#------------------------------
# Simulated data
#------------------------------

"""
    plot_truesolution(group, data::simdata, t_range, sol_group1, sol_group2; showdata=true)

Plot the ground-truth latent solution for a simulated cohort, optionally overlaying observed
(simulated) measurements.

# Arguments
- `group`: integer group selector; `1` plots `sol_group1`, otherwise `sol_group2`
- `data::simdata`: simulated dataset with groups `group1`/`group2`, `xs`, and `tvals`
- `t_range`: dense time grid (e.g., `0:0.1:10`)
- `sol_group1`, `sol_group2`: matrices of shape `(2, length(t_range))` with true latent trajectories
- `showdata::Bool=true`: overlay each subject’s observed variables as scatter

# Returns
- `curplot`: a `Plots.Plot` with the chosen true solution (and optionally data scatter).
"""
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

"""
    createindividualplot(m::odevae, data::simdata, idx::Int, sol::Matrix, trange, args::LossArgs; 
        title::String="", showtruesol::Bool=true, axislabs::Bool=false, 
        showOLS::Bool=true, colors_truesol::Array{String}=["#ff7f0e" "#1f77b4"])

Plot the learned latent trajectory for one simulated subject, optionally showing the true
latent solution and an OLS overlay.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `data::simdata`: simulated dataset with `xs`, `x_baseline`, and `tvals`
- `idx::Int`: subject index in `data`
- `sol::Matrix`: true latent solution for this subject’s group `(2, length(trange))`
- `trange`: dense time grid for plotting the smooth ODE trajectory
- `args::LossArgs`: controls smoothing (`firstonly` vs. averaged multi-IC)
- `title::String=""`: subplot title
- `showtruesol::Bool=true`: overlay the supplied true solution
- `axislabs::Bool=false`: add axis labels
- `showOLS::Bool=true`: overlay global OLS in latent space
- `colors_truesol`: two hex colors used for the true solution lines

# Returns
- `curplot`: a `Plots.Plot` object combining true solution (optional), ODE-smoothed latent means,
  encoder means (scatter), and optional OLS fit.
"""
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
            line=([:dot :dot], 4, colors_truesol)
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
    plot!(trange, smoothμs', line=(3, ["#1f77b4" "#ff7f0e"]), labels = [L"\mathrm{smooth~}\mu_1" L"\mathrm{smooth~}\mu_2"])
    if showOLS
        OLSfit = hcat(predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[1,:])))), predict(lm(@formula(Y~X), DataFrame(X=Float64.(curtvals), Y=Float64.(latentμ[2,:])))))
        plot!(curtvals, OLSfit, line = line = (3, "#e70f4f", :dash), label ="")
    end
    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}")
    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end
    plot!(title=title)
    return curplot
end

"""
    eval_z_trajectories(m::odevae, data::simdata, inds::Array{Int}, 
        sol_group1::Matrix, sol_group2::Matrix, t_range, args::LossArgs; 
        title::String="", showtruesol::Bool=true, axislabs::Bool=false, 
        showOLS::Bool=true, swapcolorcoding::Bool=false)

Display a panel of per-subject latent trajectory plots for simulated data, selecting the
appropriate ground-truth solution by group membership.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `data::simdata`: simulated dataset with `group1`, `group2`, `xs`, and `tvals`
- `inds::Array{Int}`: subject indices to visualize
- `sol_group1`, `sol_group2`: true solutions `(2, length(t_range))` for each group
- `t_range`: dense time grid for smooth curves
- `args::LossArgs`: smoothing controls (`firstonly`, etc.)
- `title::String=""`: panel title
- `showtruesol::Bool=true`: overlay true solutions
- `axislabs::Bool=false`: add axis labels in subplots
- `showOLS::Bool=true`: overlay global OLS per subject
- `swapcolorcoding::Bool=false`: swap the two colors used for the true solution

# Returns
Nothing. Displays a `Plots.Plot` panel to the current display.
"""
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

"""
    createindividualplot_piecewise(m::odevae, data::simdata, ind::Int, sol::Matrix, t_range; 
        title::String="", axislabs::Bool=false, showtruesol::Bool=true,
        showOLS::Bool=true, showglobalOLS::Bool=false,
        colors_truesol::Array{String}=["#ff7f0e" "#1f77b4"])

Plot piecewise local ODE solutions between observed time points for one simulated subject,
optionally overlaying the group’s true solution, per-segment OLS, and a global OLS fit.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `data::simdata`: simulated dataset with `xs`, `x_baseline`, and `tvals`
- `ind::Int`: subject index in `data`
- `sol::Matrix`: true solution `(2, length(t_range))` for the subject’s group
- `t_range`: dense time grid for plotting the true solution
- `title::String=""`: subplot title
- `axislabs::Bool=false`: add axis labels
- `showtruesol::Bool=true`: overlay true solution
- `showOLS::Bool=true`: overlay OLS per segment aligned to each segment start
- `showglobalOLS::Bool=false`: overlay an OLS fit over the full observed window
- `colors_truesol`: two hex colors for the true solution lines

# Returns
- `curplot`: a `Plots.Plot` object with piecewise ODE curves, uncertainty ribbons, and optional OLS overlays.
"""
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
        line=([:dot :dot], 4, colors_truesol)
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
                    line=(3, ["#1f77b4" "#ff7f0e"]), 
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
            plot!(collect(curtrange), curOLSfit, line = (3, "#e70f4f", :dash), label = label)
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
        plot!(mod_tvals, globalOLSfit, line = line = (3, "#e70f4f", :dash), label = L"\mathrm{linear~regression}")
    end

    Plots.scatter!(curtvals, latentμ[1,:], marker = (:c, 6, "#1f77b4"), label = L"\mu_1 \mathrm{~from~encoder}") 
    Plots.scatter!(curtvals, latentμ[2,:], marker = (:c, 6, "#ff7f0e"), label = L"\mu_2 \mathrm{~from~encoder}", title=title)

    if axislabs
        plot!(xlab="time in months", ylab="value of latent representation")
    end

    return curplot
end

"""
    plot_selected_ids_piecewise(m::odevae, data::simdata, inds::Array{Int}, 
        sol_group1::Matrix, sol_group2::Matrix, t_range; 
        axislabs::Bool=false, showtruesol::Bool=true, showtitle::Bool=true, 
        showOLS::Bool=true, showglobalOLS::Bool=false, 
        swapcolorcoding::Bool=false, sort_inds::Bool=true)

Create a panel of piecewise ODE plots for multiple simulated subjects. Each subject’s
true solution is chosen based on group membership.

# Arguments
- `m::odevae`: trained ODE-VAE model
- `data::simdata`: simulated dataset with `group1`, `group2`, `xs`, and `tvals`
- `inds::Array{Int}`: subject indices to visualize
- `sol_group1`, `sol_group2`: true solutions `(2, length(t_range))` for each group
- `t_range`: dense time grid for plotting the true solutions
- `axislabs::Bool=false`: add axis labels in subplots
- `showtruesol::Bool=true`: overlay true solutions
- `showtitle::Bool=true`: use the subject index as subplot title
- `showOLS::Bool=true`: overlay per-segment OLS fits
- `showglobalOLS::Bool=false`: overlay a global OLS fit per subject
- `swapcolorcoding::Bool=false`: swap colors used for true solutions
- `sort_inds::Bool=true`: sort the provided indices before plotting

# Returns
- `panelplot`: a `Plots.Plot` panel with `(length(inds)÷4, 4)` layout and customized height.
"""
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