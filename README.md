# Haskell-Math

A collection of mathematical and statistical algorithms implemented in Haskell.

## Generalized Linear Models (GLM)

This library provides functionality for fitting Generalized Linear Models (GLMs) with optional LASSO regularization. The implementation uses the Iteratively Reweighted Least Squares (IRLS) algorithm for parameter estimation.

### Features

- Support for common exponential family distributions:
  - Gaussian (Normal)
  - Bernoulli (Binomial with n=1)
  - Poisson
- Link functions:
  - Identity
  - Logit
  - Log
- LASSO (L1) regularization
- Detailed convergence tracking

### Example Usage

```haskell
import Statistics.GLM.ExponentialFamily
import Statistics.GLM.IRLS
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Create design matrix and response vector
    let x = matrix 10 3 $ \(i,j) -> if j == 0 then 1 else fromIntegral (i * j)
        y = vector [0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  -- Binary response
    
    -- Fit a logistic regression model
    result <- fitGLM x y bernoulliFamily logitLink defaultFitConfig
    
    -- Print the coefficients
    putStrLn "Estimated coefficients:"
    print $ coefficients result
```

## License

This project is licensed under the Apache License, Version 2.0.