# Data processing 

In the manuscript, we have used data from the SMArtCARE registry on patients with SMA. As the data cannot be made publicly available, we provide functions for simulating data with a similar structure.

## SMArtCARE data

```@docs
SMATestData
```

```@docs
get_SMArtCARE_data
```

```@docs
recode_SMArtCARE_data
```

## Simulated data

```@docs
simdata
```

```@docs
generate_xs
```

```@docs
generate_baseline(n, q, q_info, group1; σ_info=1.0f0, σ_noise=1.0f0)
```

```@docs
generate_baseline(n, q, q_info, group1, trueparams_group1, trueparams_group2; 
    σ_info=0.1f0, σ_noise=0.1f0)
```