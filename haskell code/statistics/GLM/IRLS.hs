{-|
Module      : Statistics.GLM.IRLS
Description : Iteratively Reweighted Least Squares for GLMs with LASSO regularization
Copyright   : (c) 2025
License     : Apache 2.0

This module implements the Iteratively Reweighted Least Squares (IRLS) algorithm
for fitting Generalized Linear Models (GLMs) with optional LASSO regularization.

The IRLS algorithm works by iteratively:
1. Computing the working responses and weights based on current estimates
2. Solving a weighted least squares problem
3. Checking for convergence

For LASSO regularization (L1 penalty), the coordinate descent algorithm is used
within each IRLS iteration to solve the penalized weighted least squares problem.
-}

module Statistics.GLM.IRLS 
( 
  -- * Types for configuring and tracking the fitting process
  FitConfig(..)
, ConvergenceInfo(..)
, FitResult(..)
  -- * Core fitting functions
, irlsUpdate
, fitGLM
, defaultFitConfig
  -- * Utilities
, initialCoefficients
, computeLogLikelihood
) where

import Statistics.GLM.ExponentialFamily
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, Vector, (#>), (<.>), (<#), fromList, toList, 
                             fromRows, toRows, diag, size, konst, atIndex)
import Control.Monad (when)
import Data.Maybe (fromMaybe)
import Text.Printf (printf)

-- | Configuration for the GLM fitting process
data FitConfig = FitConfig
  { maxIterations :: Int        -- ^ Maximum number of iterations allowed
  , tolerance :: Double         -- ^ Convergence tolerance for coefficient changes
  , llTolerance :: Double       -- ^ Convergence tolerance for log-likelihood changes
  , lambdaLasso :: Double       -- ^ LASSO regularization parameter (0 = no regularization)
  , verbose :: Bool             -- ^ Whether to print progress information
  , addIntercept :: Bool        -- ^ Whether to add an intercept term automatically
  }

-- | Default configuration for fitting
defaultFitConfig :: FitConfig
defaultFitConfig = FitConfig
  { maxIterations = 100
  , tolerance = 1e-6
  , llTolerance = 1e-8
  , lambdaLasso = 0.0
  , verbose = False
  , addIntercept = True
  }

-- | Information about the convergence process
data ConvergenceInfo = ConvergenceInfo
  { iterations :: Int              -- ^ Number of iterations performed
  , converged :: Bool              -- ^ Whether the algorithm converged
  , finalLogLikelihood :: Double   -- ^ Final log-likelihood value
  , logLikelihoodHistory :: [Double] -- ^ History of log-likelihood values
  , coefficientHistory :: [Vector Double] -- ^ History of coefficient values
  , finalDeltaNorm :: Double       -- ^ Final norm of coefficient changes
  , message :: String              -- ^ Message about convergence status
  }

-- | Result of fitting a GLM
data FitResult = FitResult
  { coefficients :: Vector Double    -- ^ Estimated coefficients
  , convergenceInfo :: ConvergenceInfo -- ^ Information about convergence
  , family :: ExponentialFamily      -- ^ Family used for fitting
  , link :: Link                     -- ^ Link function used
  , designMatrix :: Matrix Double    -- ^ Design matrix X
  , responseVector :: Vector Double  -- ^ Response vector y
  , fittedValues :: Vector Double    -- ^ Fitted values μ
  , workingWeights :: Vector Double  -- ^ Final working weights
  }

-- | Calculate initial coefficients for the GLM
-- Uses ordinary least squares for Gaussian family or
-- initializes with zeros for other families
initialCoefficients :: Matrix Double -> Vector Double -> ExponentialFamily -> Link -> Vector Double
initialCoefficients x y family link
  | familyName family == "Gaussian" = 
      let (q, r) = LA.qr x
          qty = LA.tr q LA.#> y
      in LA.fromList $ LA.toList $ LA.backSub r qty
  | otherwise = LA.konst 0 (LA.cols x)

-- | Compute linear predictor (η = Xβ)
linearPredictor :: Matrix Double -> Vector Double -> Vector Double
linearPredictor x beta = x LA.#> beta

-- | Compute fitted values (μ = g^(-1)(η))
fittedValues :: Vector Double -> Link -> Vector Double
fittedValues eta link = LA.cmap (linkInverse link) eta

-- | Compute working responses for IRLS
-- z = η + g'(μ)(y - μ)
workingResponses :: Vector Double -> Vector Double -> Vector Double -> Link -> Vector Double
workingResponses y mu eta link = 
    let deriv = LA.cmap (linkDerivative link) mu
        adjustment = LA.zipVectorWith (*) deriv (LA.zipVectorWith (-) y mu)
    in LA.zipVectorWith (+) eta adjustment

-- | Compute working weights for IRLS
-- W = 1 / (g'(μ)² * V(μ))
workingWeights :: Vector Double -> Link -> ExponentialFamily -> Vector Double
workingWeights mu link family =
    let deriv = LA.cmap (linkDerivative link) mu
        variance = LA.cmap (cumulantSecondDerivative family) (LA.cmap (linkFunction link) mu)
        denominators = LA.zipVectorWith (*) (LA.cmap (^2) deriv) variance
    in LA.cmap (\x -> if x > 1e-10 then 1/x else 1e10) denominators

-- | Compute log-likelihood for the current model
computeLogLikelihood :: Vector Double -> Vector Double -> ExponentialFamily -> Link -> Double
computeLogLikelihood y mu family _ =
    let theta = LA.cmap (linkFunction (canonicalLink family)) mu
        logLiks = LA.zipVectorWith (logLikelihood family) y theta
    in LA.sumElements logLiks

-- | Soft threshold function for LASSO coordinate descent
softThreshold :: Double -> Double -> Double
softThreshold lambda x
  | x > lambda = x - lambda
  | x < -lambda = x + lambda
  | otherwise = 0

-- | Coordinate descent step for solving LASSO-regularized weighted least squares
coordinateDescentStep :: Matrix Double -> Vector Double -> Vector Double -> Double -> Vector Double -> Vector Double
coordinateDescentStep x z w lambda beta =
    let p = LA.cols x
        doCoordinate j beta' =
            let xj = LA.takeColumn x j
                wXj = LA.zipVectorWith (*) w xj
                updatedBeta = LA.subVector 0 p beta'
                residual = LA.zipVectorWith (-) z (x LA.#> updatedBeta)
                wRj = LA.dotVector wXj residual + (LA.dotVector wXj xj) * (beta' `atIndex` j)
                coef = wRj / (LA.dotVector wXj xj)
            in LA.accum beta' j (const (softThreshold lambda coef))
    in foldl (\b j -> doCoordinate j b) beta [0..(p-1)]

-- | One IRLS update iteration
-- Returns updated coefficients and convergence information
irlsUpdate :: Vector Double -> Matrix Double -> Vector Double -> ExponentialFamily -> Link -> Double -> 
             (Vector Double, Double, Double)
irlsUpdate beta x y family link lambda = 
    let eta = linearPredictor x beta
        mu = fittedValues eta link
        z = workingResponses y mu eta link
        w = workingWeights mu link family
        
        -- Solve weighted least squares with LASSO penalty using coordinate descent
        maxCdIters = 50
        cdTol = 1e-8
        
        cdIteration beta' tol iter
            | iter >= maxCdIters = beta'
            | otherwise = 
                let beta'' = coordinateDescentStep x z w lambda beta'
                    delta = LA.norm_2 (LA.zipVectorWith (-) beta'' beta')
                in if delta < tol
                   then beta''
                   else cdIteration beta'' tol (iter + 1)
        
        betaNew = cdIteration beta cdTol 0
        
        -- Calculate log-likelihood for new coefficients
        etaNew = linearPredictor x betaNew
        muNew = fittedValues etaNew link
        ll = computeLogLikelihood y muNew family link
        
        -- Calculate change in coefficients
        deltaNorm = LA.norm_2 (LA.zipVectorWith (-) betaNew beta)
    
    in (betaNew, ll, deltaNorm)

-- | Main function to fit a GLM using IRLS
fitGLM :: Matrix Double -> Vector Double -> ExponentialFamily -> Link -> FitConfig -> IO FitResult
fitGLM xRaw yRaw family link config = do
    -- Validate inputs
    let n = LA.rows xRaw
        p = LA.cols xRaw
        
    when (n /= LA.size yRaw) $
        error $ "Design matrix has " ++ show n ++ " rows, but response vector has " 
                ++ show (LA.size yRaw) ++ " elements"
    
    -- Check response values
    let validResponses = all (validateResponse family) (LA.toList yRaw)
    when (not validResponses) $
        error $ "Response vector contains invalid values for " ++ familyName family ++ " family"
    
    -- Add intercept if requested
    let x = if addIntercept config
            then LA.fromRows $ map (\row -> 1 LA.<| row) (LA.toRows xRaw)
            else xRaw
        y = yRaw
        
    -- Initialize coefficients
    let beta0 = initialCoefficients x y family link
        mu0 = fittedValues (linearPredictor x beta0) link
        ll0 = computeLogLikelihood y mu0 family link
    
    when (verbose config) $
        putStrLn $ "Starting IRLS with initial log-likelihood: " ++ show ll0
    
    -- Iteratively update using IRLS
    let 
        iterate beta prevLL iter llHistory betaHistory
            | iter >= maxIterations config = 
                let info = ConvergenceInfo 
                        { iterations = iter
                        , converged = False
                        , finalLogLikelihood = prevLL
                        , logLikelihoodHistory = reverse (prevLL:llHistory)
                        , coefficientHistory = reverse (beta:betaHistory)
                        , finalDeltaNorm = 0.0
                        , message = "Maximum iterations reached without convergence"
                        }
                in return $ FitResult 
                        { coefficients = beta
                        , convergenceInfo = info
                        , family = family
                        , link = link
                        , designMatrix = x
                        , responseVector = y
                        , fittedValues = fittedValues (linearPredictor x beta) link
                        , workingWeights = workingWeights (fittedValues (linearPredictor x beta) link) link family
                        }
            | otherwise = do
                when (verbose config && iter `mod` 5 == 0) $
                    putStrLn $ "Iteration " ++ show iter ++ ", Log-likelihood: " ++ show prevLL
                
                let (newBeta, newLL, deltaNorm) = irlsUpdate beta x y family link (lambdaLasso config)
                    llChange = abs (newLL - prevLL)
                    
                let newBetaHistory = beta : betaHistory
                    newLLHistory = prevLL : llHistory
                
                if deltaNorm < tolerance config && llChange < llTolerance config
                then do
                    when (verbose config) $ do
                        putStrLn $ "Converged after " ++ show (iter + 1) ++ " iterations"
                        putStrLn $ "Final log-likelihood: " ++ show newLL
                        putStrLn $ "Coefficient change norm: " ++ show deltaNorm
                    
                    let info = ConvergenceInfo 
                            { iterations = iter + 1
                            , converged = True
                            , finalLogLikelihood = newLL
                            , logLikelihoodHistory = reverse (newLL:newLLHistory)
                            , coefficientHistory = reverse (newBeta:newBetaHistory)
                            , finalDeltaNorm = deltaNorm
                            , message = "Converged successfully"
                            }
                            
                    return $ FitResult 
                            { coefficients = newBeta
                            , convergenceInfo = info
                            , family = family
                            , link = link
                            , designMatrix = x
                            , responseVector = y
                            , fittedValues = fittedValues (linearPredictor x newBeta) link
                            , workingWeights = workingWeights (fittedValues (linearPredictor x newBeta) link) link family
                            }
                else
                    iterate newBeta newLL (iter + 1) newLLHistory newBetaHistory
    
    -- Start the iteration
    iterate beta0 ll0 0 [] []
