module LatentDynamics 

using GLM 
using Distributions 
using Flux
using LaTeXStrings
using Plots
using Parameters

include("PreprocessingSMArtCARE.jl")
include("SimulateData.jl")
include("Model.jl")
include("Plotting.jl")
include("Evaluate.jl")

export 
    get_SMArtCARE_data, recode_SMArtCARE_data,
    simdata, generate_xs, generate_baseline, 
    generalsolution, 
    params_fullinhomogeneous, params_offdiagonalinhomogeneous, params_diagonalinhomogeneous, 
    params_driftonly, params_fullhomogeneous, params_diagonalhomogeneous, 
    odevae, ModelArgs, LossArgs,
    loss, train_model!, 
    SMATestData, 
    createindividualplot, createindividualplot_piecewise, 
    plot_selected_ids, plot_selected_ids_piecewise,
    plot_truesolution, eval_z_trajectories,
    eval_prediction, eval_reconstructed_prediction
end