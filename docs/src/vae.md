# ODE-VAE model 

We integrate the ODE-based dynamic model in the latent space of a VAE for flexible non-linear dimension reduction, reflecting the assumption of a lower-dimensional underlying dynamic process driving the observed measurements. The following functions are used to define, construct and train an ODE-VAE model. 

To jointly optimize all components, i.e., the dynamic model and the VAE for dimension reduction, and the ODE-net for obtaining person-specific ODE parameters, we implement a joint loss function that incorporates all components and optimize it by stochastic gradient descent. This requires to differentiate through our ODE estimator and the calculation of time-dependent inverse-variance weights. Here, we exploit the flexible automatic differentiation system from `Zygote.jl` to simultaneously obtain gradients with respect to the VAE encoder and decoder parameters and the individual-specific dynamic model parameters in a straightforward way that requires minimal code adaptation. Zygote is specifically useful for that because of its very powerful source-to-source differentiation, that allows for differentiate through arbitrary Julia code, including the ODE solvers, user-defined structs, loops and recursion without any code refactoring or adaptation. For details, check out, e.g., [Innes et al. (2019)](https://arxiv.org/abs/1907.07587).

As a result of this joint optimization, the components can influence each other, such that a latent representation can be found that is automatically adapted to the underlying dynamics and the ODE system structures and regularizes the representation. 

As our ODEs have analytical solutions, differentiation through the latent dynamics estimator does not require backpropagating gradients through a numerical ODE solving step. However, differentiable programming also allows for differentiating through ODE solvers in each loss function gradient update, which can be realized efficiently, e.g., using the [adjoint sensitivity method](https://arxiv.org/abs/1806.07366).

## Defining the model 

```@docs
odevae
```

```@docs
odevae(modelargs::ModelArgs)
```

```@docs
ModelArgs
```

## Training the model

```@docs
LossArgs
```

```@docs
loss
```

```@docs
train_model!
```