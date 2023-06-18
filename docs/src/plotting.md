# Plotting 

We provide various functions to visualize the learnt latent trajectories, both for individual patients and panels of multiple patients, for both simulated data and SMArtCARE data. 
We provide functions to plot the trajectories of the overall estimator of the solution and functions to plot the prediction from the model starting at the current time point until the next time point (the `_piecewise` functions).

## Simulated data

```@docs
plot_truesolution
```

### Individual trajectories

```@docs
createindividualplot(m::odevae, 
        data::simdata, 
        idx::Int, 
        sol::Matrix, 
        trange, 
        args::LossArgs;
        title::String="", 
        showtruesol::Bool=true,
        axislabs::Bool=false, 
        showOLS::Bool=true, 
        colors_truesol::Array{String} = ["#ff7f0e" "#1f77b4"]
)
```

```@docs
createindividualplot_piecewise(m::odevae, 
        data::simdata, 
        ind::Int, 
        sol::Matrix, 
        t_range; 
        title::String="", 
        axislabs::Bool=false, 
        showtruesol::Bool=true,
        showOLS::Bool=true, 
        showglobalOLS::Bool=false,
        colors_truesol::Array{String} = ["#ff7f0e" "#1f77b4"]
)
```

```@docs
eval_z_trajectories
```

### Panels of trajectories 

```@docs
plot_seleced_ids_piecewise(m::odevae, 
        data::simdata, 
        inds::Array{Int}, 
        sol_group1::Matrix, 
        sol_group2::Matrix, 
        t_range; 
        axislabs::Bool=false, 
        showtruesol::Bool=true,
        showtitle::Bool=true, 
        showOLS::Bool=true, 
        showglobalOLS::Bool=false,
        swapcolorcoding::Bool=false, 
        sort_inds::Bool=true
    )
```

## SMArtCARE data 

### Individual trajectories

```@docs
createindividualplot(m::odevae, 
        testdata::SMATestData, 
        args::LossArgs, 
        patient_id; 
        axislabs::Bool=false, 
        title::String="", 
        showOLS::Bool=true
)
```

```@docs
createindividualplot_piecewise(m::odevae, 
        testdata::SMATestData, 
        patient_id; 
        title::String="", 
        showOLS::Bool=true, 
        axislabs::Bool=false
)
```

### Panels of trajectories 

```@docs
plot_selected_ids
```

```@docs
plot_selected_ids_piecewise(m::odevae, 
    testdata::SMATestData,
    selected_ids::Array; 
    showOLS::Bool=true
)
```


