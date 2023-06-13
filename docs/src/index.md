# LatentDynamics.jl

![](assets/repo_intro_simulation.jpg)

Julia package for fitting dynamic models based on ODEs in the latent space of a VAE. Specifically, it implements an approach where each observation serves as the initial value to obtain multiple local ODE solutions which are used to build an inverse-variance weighted estimator of the underlying dynamics. This reduces dependence of the ODE solution on the initial condition and can provide more robust estimation of the underlying trajectory particularly in settings with a high level of noicse. To deal with a larger number of variables, the approach is combined with a VAE for dimension reduction, as the ODE systems are defined and solved in the latent space of a VAE model.
The model has been developed for the setting of a clinical registry, e.g., from a rare disease, where data might be noisy and heterogeneous and only few and irregular time points are available per patients. In this setting, we use the characterization of patients at the baseline time point to learn individual trajectories, by mapping each patient's baseline variables with an additional neural network to a set of individual-specific ODE parameters. We simultaneously optimize all components, the VAE for dimension reduction, the dynamic model and the network for mapping baseline variables to ODE parameters, using differentiable programming. This allows for finding a low-dimensional representation that is specifically adapted to the underlying trajectories as described by the person-specific ODE systems. 

The package includes functions for defining and training the VAE with and ODE-based dynamic in latent space, including a wide range of customizable hyperparameters and options for controlling the training behavior. There are also different choices for the underlying ODE system with different numbers of parameters. Further, the package provides functions for visualizing the learned latent trajectories and evaluating prediction performance at subsequent time points, both in latent space and on the reconstructed data and in comparison to simpler baseline models. In addition, the package provides functions for loading and pre-processing data from the SMArtCARE registry on spinal muscular atrophy (SMA) patients, which is used as an example application in the corresponding manuscript, and, as this data is not publicly available, for simulating data with a similar structure. 

For more details, please have a look at our manuscript [Hackenberg et al. (2023) A statistical approach to latent dynamic modeling with differential equations](arXiv_link).


## Contents

```@contents 
Pages = [
    "data.md", 
    "dynamics.md",
    "vae.md", 
    "optimization.md", 
    "plotting.md", 
    "evaluation.md"
]
```