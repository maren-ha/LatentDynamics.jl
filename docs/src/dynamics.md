# Dynamic model 

We use a dynamic model based on ODEs, fitted in a statistical framework. Specifically, we solve multiple ODEs, using each observed value as the initial value and averaging the solutions using time-dependent inverse-variance weights. We thus obtain to obtain an unbiased minimum variance estimator of the true underlying dynamics. With this approach, we reduce dependence on the initial condition and increase robustness to noise, yet maintain a local perspective and mechanistic interpretation. 

For modeling with a larger number of variables, we integrate the dynamic model in the latent space of a VAE for flexible non-linear dimension reduction, reflecting the assumption of a lower-dimensional underlying dynamic process driving the observed measurements. 

```@docs
generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32}) 
```

```@docs
generalsolution(t, x0::Vector{Float32}, c::Vector{Float32}) 
```

```@docs
params_fullinhomogeneous
```

```@docs
params_offdiagonalinhomogeneous
```

```@docs
params_diagonalinhomogeneous
```

```@docs
params_fullhomogeneous
```

```@docs
params_diagonalhomogeneous
```

```@docs
params_driftonly
```