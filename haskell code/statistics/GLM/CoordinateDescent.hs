{-|
Module      : Statistics.GLM.CoordinateDescent
Description : Pure coordinate descent solver for LASSO-regularized weighted least squares
Copyright   : (c) 2025
License     : Apache 2.0

This module implements a coordinate descent algorithm for solving LASSO-regularized
weighted least squares problems that arise in GLM fitting with L1 regularization.

The objective function to be minimized is:

    f(β) = || W^(1/2) (z - Xβ) ||_2^2 + λ ||β||_1

where:
  - X is the design matrix
  - z is the pseudo-response vector
  - W is a diagonal matrix of working weights
  - λ is the LASSO penalty parameter
  - β is the coefficient vector

Reference:
  Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for 
  generalized linear models via coordinate descent. Journal of statistical 
  software, 33(1), 1.
-}

module Statistics.GLM.CoordinateDescent (
  -- * Main solver
  coordinateDescentSolver,
  getCoefficientsOnly,
  
  -- * Types for tracking convergence
  CDConvergenceInfo(..),
  CDResult(..),
  
  -- * Helper functions
  softThreshold,
  updateCoordinate,
  computeObjective,
  coordinateDescent
) where

import Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra.Devel as LA (axpy)
import Debug.Trace (trace)
import Text.Printf (printf)
import Control.Monad (when)
import Data.List (foldl')

zipVectorWith :: (Double -> Double -> Double) -> Vector Double -> Vector Double -> Vector Double
zipVectorWith f v1 v2 = fromList $ zipWith f (toList v1) (toList v2)

-- | Information about the coordinate descent convergence process
data CDConvergenceInfo = CDConvergenceInfo
  { cdIterations :: Int             -- ^ Number of iterations performed
  , cdConverged :: Bool             -- ^ Whether the algorithm converged
  , cdFinalObjective :: Double      -- ^ Final objective function value
  , cdObjectiveHistory :: [Double]  -- ^ History of objective function values
  , cdMaxDeltaHistory :: [Double]   -- ^ History of maximum coefficient changes
  , cdFinalMaxDelta :: Double       -- ^ Final maximum coefficient change
  , cdMessage :: String             -- ^ Message about convergence status
  }

-- | Result of the coordinate descent algorithm
data CDResult = CDResult
  { cdCoefficients :: Vector Double      -- ^ Final coefficient estimates
  , cdConvergenceInfo :: CDConvergenceInfo -- ^ Convergence information
  }

-- | Soft-thresholding operator used for L1 regularization
--
-- S(x, λ) = sign(x) * max(|x| - λ, 0)
--
-- This operator induces sparsity by shrinking coefficients toward zero and
-- setting small coefficients exactly to zero when |x| <= λ.
softThreshold :: Double -> Double -> Double
softThreshold lambda x
  | x > lambda = x - lambda
  | x < (-lambda) = x + lambda
  | otherwise = 0.0

-- | Update a single coefficient using the coordinate descent update rule
--
-- For coefficient j, the update is:
--
-- β_j = S(R_j / (X_j^T W X_j), λ / (X_j^T W X_j))
--
-- where R_j = X_j^T W (z - X_{-j}β_{-j}) is the weighted correlation of feature j
-- with the residual (ignoring the contribution of feature j).
--
-- This can be computed efficiently as:
-- R_j = X_j^T W z - X_j^T W X_{-j}β_{-j}
--     = X_j^T W z - ∑_{k≠j} (X_j^T W X_k)β_k
--
-- When working with the current residual r = z - Xβ, we can use:
-- R_j = X_j^T W (r + X_j β_j)
updateCoordinate :: Matrix Double -> Vector Double -> Vector Double -> Vector Double -> Double -> Int -> Vector Double -> Vector Double
updateCoordinate x z w residual lambda j beta = 
  let xj = x ? [j]  -- j-th column of X
      wXj = zipVectorWith (*) w xj
      
      -- Calculate weighted dot product of feature j with itself
      denominator = dot wXj xj
      
      -- Safety check for numerical stability
      safeDiv = if denominator > 1e-10 then 1.0 / denominator else 1e10
      
      -- Current coefficient value
      betaJ = beta ! j
      
      -- Correlation of feature j with residual plus its own contribution
      rPlusOwn = zipVectorWith (+) residual (scale betaJ xj)
      correlation = dot wXj rPlusOwn
      
      -- Coordinate update with soft thresholding
      newBetaJ = softThreshold (lambda * safeDiv) (correlation * safeDiv)
      
      -- Update the coefficient vector
      newBeta = beta
      newBeta' = if betaJ /= newBetaJ 
                 then accum newBeta [(j, newBetaJ)]
                 else newBeta
  in newBeta'

-- | Calculate updated residual after coefficient update
--
-- r' = r + X_j (β_j^old - β_j^new)
updateResidual :: Vector Double -> Matrix Double -> Int -> Double -> Double -> Vector Double
updateResidual residual x j oldBeta newBeta =
  let xj = x ? [j]
      diff = oldBeta - newBeta
  in axpy diff xj residual  -- r' = r + diff * xj

-- | Compute the objective function value
--
-- f(β) = || W^(1/2) (z - Xβ) ||_2^2 + λ ||β||_1
computeObjective :: Matrix Double -> Vector Double -> Vector Double -> Double -> Vector Double -> Double
computeObjective x z w lambda beta =
  let residual = z - (x #> beta)
      weightedResidual = zipVectorWith (*) (sqrt <$> w) residual
      squaredError = dot weightedResidual weightedResidual
      l1Norm = sumElements (cmap abs beta)
  in squaredError + lambda * l1Norm

-- | Main coordinate descent solver for LASSO-regularized weighted least squares.
--
-- This function implements the coordinate descent algorithm to solve:
--
--   min_β { || W^(1/2) (z - Xβ) ||_2^2 + λ ||β||_1 }
--
-- The algorithm iteratively updates each coefficient while holding others fixed,
-- continuing until convergence or maximum iterations are reached.
--
-- Parameters:
--   * x: Design matrix X.
--   * z: Pseudo-response vector.
--   * w: Vector of working weights.
--   * lambda: LASSO penalty parameter.
--   * beta0: Initial coefficient estimates.
--   * maxIter: Maximum number of iterations (default: 1000).
--   * tol: Convergence tolerance for coefficient changes (default: 1e-6).
--   * verbose: Whether to print progress information (default: False).
--
-- Returns:
--   CDResult containing final coefficients and convergence information.
coordinateDescentSolver :: Matrix Double -> Vector Double -> Vector Double -> Double -> Vector Double -> 
                          Int -> Double -> Bool -> CDResult
coordinateDescentSolver x z w lambda beta0 maxIter tol verbose = 
  -- Validate inputs
  if (rows x /= size z) || (rows x /= size w) || (cols x /= size beta0)
    then error $ printf 
      "Dimension mismatch: X(%d,%d), z(%d), w(%d), beta0(%d)" 
      (rows x) (cols x) (size z) (size w) (size beta0)
    else if any (< 0) (toList w)
      then error "Weights must be non-negative"
      else
        -- Initialize solver
        let n = rows x
            p = cols x
            initialResidual = z - (x #> beta0)
            initialObj = computeObjective x z w lambda beta0
            
            -- Inner loop: one complete cycle through all coordinates
            cycleUpdate beta residual iter objHistory deltaHistory =
              let -- Update each coordinate in sequence
                  (newBeta, newResidual, changes) = 
                    foldl' updateCoord (beta, residual, []) [0..(p-1)]
                  
                  -- Helper to update a single coordinate and track changes
                  updateCoord (b, r, deltas) j = 
                    let oldBeta = b ! j
                        b' = updateCoordinate x z w r lambda j b
                        newBeta = b' ! j
                        r' = if oldBeta /= newBeta
                             then updateResidual r x j oldBeta newBeta
                             else r
                        delta = abs (newBeta - oldBeta)
                    in (b', r', delta:deltas)
                  
                  -- Calculate objective value and check for convergence
                  newObj = computeObjective x z w lambda newBeta
                  maxDelta = maximum (0:changes)  -- Use 0 as fallback
                  converged = maxDelta < tol
                  
                  -- Updated histories
                  newObjHistory = newObj : objHistory
                  newDeltaHistory = maxDelta : deltaHistory
              in 
                -- Progress report in verbose mode
                if verbose && (iter `mod` 10 == 0)
                  then trace (printf "CD Iteration %d, max change: %.6f, objective: %.6f" 
                             iter maxDelta newObj) 
                       (newBeta, newResidual, converged, newObjHistory, newDeltaHistory)
                  else (newBeta, newResidual, converged, newObjHistory, newDeltaHistory)
            
            -- Outer loop: repeat cycles until convergence or max iterations
            iterate beta residual iter objHistory deltaHistory
              | iter >= maxIter = 
                  let finalObj = head objHistory
                      finalDelta = head deltaHistory
                      info = CDConvergenceInfo
                        { cdIterations = iter
                        , cdConverged = False
                        , cdFinalObjective = finalObj
                        , cdObjectiveHistory = reverse objHistory
                        , cdMaxDeltaHistory = reverse deltaHistory
                        , cdFinalMaxDelta = finalDelta
                        , cdMessage = "Maximum iterations reached without convergence"
                        }
                  in CDResult
                        { cdCoefficients = beta
                        , cdConvergenceInfo = info
                        }
              | otherwise = 
                  let (beta', residual', converged, newObjHistory, newDeltaHistory) = 
                        cycleUpdate beta residual iter objHistory deltaHistory
                  in if converged
                      then 
                        let finalObj = head newObjHistory
                            finalDelta = head newDeltaHistory
                            info = CDConvergenceInfo
                              { cdIterations = iter + 1
                              , cdConverged = True
                              , cdFinalObjective = finalObj
                              , cdObjectiveHistory = reverse newObjHistory
                              , cdMaxDeltaHistory = reverse newDeltaHistory
                              , cdFinalMaxDelta = finalDelta
                              , cdMessage = "Converged successfully"
                              }
                        in CDResult
                              { cdCoefficients = beta'
                              , cdConvergenceInfo = info
                              }
                      else iterate beta' residual' (iter + 1) newObjHistory newDeltaHistory
            
        in iterate beta0 initialResidual 0 [initialObj] [1.0]  -- Start with arbitrary initial delta

-- | Helper function to get only coefficients from the solver
-- For backward compatibility or when convergence info isn't needed
getCoefficientsOnly :: Matrix Double -> Vector Double -> Vector Double -> Double -> Vector Double -> 
                      Int -> Double -> Bool -> Vector Double
getCoefficientsOnly x z w lambda beta0 maxIter tol verbose =
  cdCoefficients $ coordinateDescentSolver x z w lambda beta0 maxIter tol verbose

-- | Default implementation with standard parameters
--
-- This is a convenience wrapper around the full coordinateDescentSolver function.
coordinateDescentSolver' :: Matrix Double -> Vector Double -> Vector Double -> Double -> Vector Double -> Vector Double
coordinateDescentSolver' x z w lambda beta0 = 
  getCoefficientsOnly x z w lambda beta0 1000 1e-6 False

coordinateDescent :: Int -> Int
coordinateDescent x = x + 1
