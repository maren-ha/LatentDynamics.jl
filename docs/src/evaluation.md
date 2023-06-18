# Evaluation 

## Simulated data 

```@docs
eval_prediction(m::odevae, data::simdata, n_future_tps::Int=1)
```

```@docs
eval_reconstructed_prediction(m::odevae, data::simdata, n_future_tps::Int=1)
```

## SMArtCARE data 

```@docs
eval_prediction(m::odevae, testdata::SMATestData, n_future_tps::Int=1)
```

```@docs
eval_reconstructed_prediction(m::odevae, testdata::SMATestData, n_future_tps::Int=1)
```
