{-|
Module      : Statistics.GLM.ExponentialFamily
Description : Exponential family distributions for GLMs with LASSO regularization
Copyright   : (c) 2025
License     : Apache 2.0

This module provides implementations of exponential family distributions for use
in Generalized Linear Models (GLMs). The exponential family has the form:

    f(y|θ) = h(y) * exp(η(θ) * T(y) - A(θ))

Where:
  - θ is the canonical parameter
  - η(θ) is the natural parameter
  - T(y) is the sufficient statistic
  - A(θ) is the log-partition (cumulant) function
  - h(y) is the base measure
-}

module Statistics.GLM.ExponentialFamily where

import Data.Maybe (fromMaybe)

-- | Link function between mean parameter and natural parameter
data Link = Link 
  { linkFunction :: Double -> Double       -- ^ g(μ): maps mean to linear predictor
  , linkDerivative :: Double -> Double     -- ^ g'(μ): derivative of link function
  , linkInverse :: Double -> Double        -- ^ g^(-1)(η): maps linear predictor to mean
  , linkName :: String                     -- ^ Name of the link function
  }

-- | Exponential family distribution representation
data ExponentialFamily = ExponentialFamily
  { familyName :: String                    -- ^ Name of the distribution
  , logLikelihood :: Double -> Double -> Double  -- ^ log f(y|θ): log-likelihood function
  , cumulant :: Double -> Double            -- ^ A(θ): cumulant/log-partition function
  , cumulantDerivative :: Double -> Double  -- ^ A'(θ): mean function μ(θ)
  , cumulantSecondDerivative :: Double -> Double  -- ^ A''(θ): variance function V(θ)
  , sufficientStatistic :: Double -> Double -- ^ T(y): sufficient statistic
  , canonicalLink :: Link                   -- ^ The canonical link function for this family
  , validateResponse :: Double -> Bool      -- ^ Checks if a response value is valid
  }

-- | Identity link function: g(μ) = μ
-- Used as canonical link for Gaussian family
identityLink :: Link
identityLink = Link
  { linkFunction = id
  , linkDerivative = const 1
  , linkInverse = id
  , linkName = "identity"
  }

-- | Log link function: g(μ) = log(μ)
-- Used as canonical link for Poisson family
logLink :: Link
logLink = Link
  { linkFunction = log
  , linkDerivative = recip
  , linkInverse = exp
  , linkName = "log"
  }

-- | Logit link function: g(μ) = log(μ/(1-μ))
-- Used as canonical link for Bernoulli family
logitLink :: Link
logitLink = Link
  { linkFunction = \mu -> log (mu / (1 - mu))
  , linkDerivative = \mu -> 1 / (mu * (1 - mu))
  , linkInverse = \eta -> let expEta = exp eta in expEta / (1 + expEta)
  , linkName = "logit"
  }

-- | Bernoulli distribution for binary outcomes (0, 1)
-- Probability mass function: p(y) = μ^y * (1-μ)^(1-y)
bernoulliFamily :: ExponentialFamily
bernoulliFamily = ExponentialFamily
  { familyName = "Bernoulli"
  , logLikelihood = \y theta -> y * theta - log (1 + exp theta)
  , cumulant = \theta -> log (1 + exp theta)
  , cumulantDerivative = \theta -> 
      let expTheta = exp theta 
      in expTheta / (1 + expTheta)  -- sigmoid function
  , cumulantSecondDerivative = \theta -> 
      let expTheta = exp theta
          mu = expTheta / (1 + expTheta)
      in mu * (1 - mu)  -- variance is μ(1-μ)
  , sufficientStatistic = id
  , canonicalLink = logitLink
  , validateResponse = \y -> y == 0 || y == 1
  }

-- | Poisson distribution for count data
-- Probability mass function: p(y) = μ^y * e^(-μ) / y!
poissonFamily :: ExponentialFamily
poissonFamily = ExponentialFamily
  { familyName = "Poisson"
  , logLikelihood = \y theta -> y * theta - exp theta - logFactorial y
  , cumulant = exp
  , cumulantDerivative = exp
  , cumulantSecondDerivative = exp  -- variance equals mean for Poisson
  , sufficientStatistic = id
  , canonicalLink = logLink
  , validateResponse = \y -> y >= 0 && fromInteger (round y) == y  -- non-negative integer
  }
  where
    logFactorial :: Double -> Double
    logFactorial n
      | n <= 1    = 0
      | otherwise = sum $ map log [1..n]  -- More efficient implementation possible

-- | Gaussian (Normal) distribution for continuous outcomes
-- Probability density function: p(y) = (1/√(2πσ²)) * exp(-(y-μ)²/(2σ²))
-- Note: This implementation assumes unit dispersion (σ² = 1)
gaussianFamily :: ExponentialFamily
gaussianFamily = ExponentialFamily
  { familyName = "Gaussian"
  , logLikelihood = \y theta -> -(y - theta)^2 / 2 - log (2 * pi) / 2
  , cumulant = \theta -> theta^2 / 2
  , cumulantDerivative = id
  , cumulantSecondDerivative = const 1  -- variance is constant (=1) for standard Gaussian
  , sufficientStatistic = id
  , canonicalLink = identityLink
  , validateResponse = const True  -- all real values are valid
  }

-- | Fit a GLM with LASSO regularization
-- This is a placeholder - you'd implement the actual fitting algorithm
fitGLM :: ExponentialFamily -> Link -> [[Double]] -> [Double] -> Double -> IO [Double]
fitGLM family link features targets lambda = do
  -- Placeholder implementation
  -- A real implementation would use coordinate descent or similar algorithm
  -- to minimize the penalized negative log-likelihood
  putStrLn $ "Fitting GLM with " ++ familyName family ++ " family"
  putStrLn $ "Using " ++ linkName link ++ " link function"
  putStrLn $ "LASSO regularization parameter: " ++ show lambda
  
  -- Return placeholder coefficients
  return $ replicate (length (head features) + 1) 0.0
