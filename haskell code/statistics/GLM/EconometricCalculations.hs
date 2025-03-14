{-|
Module      : Statistics.GLM.EconometricCalculations
Description : Econometric calculations for GLM, including Fisher information and confidence intervals
Copyright   : (c) 2025
License     : Apache 2.0

This module provides functions for econometric calculations after fitting a 
Generalized Linear Model (GLM), including computation of the Fisher information matrix,
coefficient variance-covariance matrix, confidence intervals, and diagnostic reports.
-}

module Statistics.GLM.EconometricCalculations 
( 
  -- * Fisher Information Matrix
  fisherInformation
, pseudoInverse
  -- * Variance-Covariance Matrix
, varianceCovarianceMatrix  
  -- * Confidence Intervals
, standardErrors
, confidenceIntervals
, criticalValue
  -- * Diagnostic Reports
, DiagnosticReport(..)
, diagnosticReport
) where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, Vector, (<>))
import Control.Exception (assert)
import Text.Printf (printf)

-- | Compute the Fisher Information Matrix: I(β) = X^T W X
-- where X is the design matrix and W is a diagonal matrix of weights.
-- 
-- The Fisher Information Matrix gives the amount of information that the
-- observable variables carry about the parameters of the distribution.
fisherInformation :: Matrix Double  -- ^ Design matrix X
                 -> Vector Double   -- ^ Vector of weights (diagonal of W)
                 -> Matrix Double   -- ^ Fisher Information Matrix
fisherInformation x w = 
    -- Validate dimensions
    if LA.rows x /= LA.size w
    then error $ "fisherInformation: Design matrix rows must match weights vector length" 
                 ++ " (got " ++ show (LA.rows x) ++ " and " ++ show (LA.size w) ++ ")"
    else
        -- Compute X^T W X
        let wSqrt = LA.cmap sqrt w  -- Square root of weights
            xw = LA.diagR 0 wSqrt LA.<> x  -- Equivalent to W^(1/2) X
        in LA.tr xw LA.<> xw  -- (W^(1/2) X)^T (W^(1/2) X) = X^T W X

-- | Compute a pseudo-inverse of a matrix using SVD.
-- This is more numerically stable than direct inversion for
-- potentially ill-conditioned matrices.
pseudoInverse :: Matrix Double  -- ^ Matrix to invert
             -> Matrix Double   -- ^ Pseudo-inverse
pseudoInverse mat = 
    let (u, s, v) = LA.svd mat
        tol = 1e-10 * maximum (LA.toList s)  -- Numerical stability threshold
        sInv = LA.vector [if x > tol then 1/x else 0 | x <- LA.toList s]
        sInvMat = LA.diag sInv
    in v LA.<> sInvMat LA.<> LA.tr u

-- | Compute the variance-covariance matrix for the estimated coefficients.
-- This is the inverse of the Fisher Information Matrix.
varianceCovarianceMatrix :: Matrix Double  -- ^ Fisher Information Matrix
                        -> Matrix Double   -- ^ Variance-covariance matrix
varianceCovarianceMatrix fim =
    -- Check if matrix is invertible
    if LA.rank fim == min (LA.rows fim) (LA.cols fim)
    then 
        -- Try standard inversion first
        let maybeInv = LA.inv fim
        in if LA.diagL 0 (LA.abs maybeInv) LA.>= LA.scalar 1e-10
           then maybeInv -- Use standard inverse if well-conditioned
           else pseudoInverse fim  -- Use pseudo-inverse for numerical stability
    else pseudoInverse fim  -- Use pseudo-inverse for rank-deficient matrix

-- | Compute standard errors for the coefficient estimates.
-- Standard errors are the square roots of the diagonal elements of
-- the variance-covariance matrix.
standardErrors :: Matrix Double  -- ^ Variance-covariance matrix
              -> Vector Double   -- ^ Standard errors for each coefficient
standardErrors vcov = LA.cmap sqrt $ LA.takeDiag vcov

-- | Calculate the critical value for a given confidence level.
-- For example, a 95% confidence level corresponds to a critical value of 1.96.
-- This implementation uses the normal approximation.
criticalValue :: Double  -- ^ Confidence level (between 0 and 1)
             -> Double   -- ^ Critical value
criticalValue confLevel = 
    -- Validate input
    if not (confLevel > 0 && confLevel < 1)
    then error $ "criticalValue: Confidence level must be between 0 and 1, got " ++ show confLevel
    else
        -- Critical value for two-tailed test
        let alpha = 1 - confLevel
            -- This is an approximation; ideally use statistical libraries
            -- for more accurate calculation of quantiles
        in case () of
             _ | confLevel >= 0.99 -> 2.576  -- 99% CI
               | confLevel >= 0.98 -> 2.326  -- 98% CI
               | confLevel >= 0.95 -> 1.96   -- 95% CI
               | confLevel >= 0.90 -> 1.645  -- 90% CI
               | confLevel >= 0.80 -> 1.282  -- 80% CI
               | otherwise -> 1.0           -- Default fallback

-- | Calculate confidence intervals for coefficient estimates.
-- For each coefficient β_i, the confidence interval is:
-- β_i ± z × sqrt(Var(β_i))
-- where z is the critical value for the desired confidence level.
confidenceIntervals :: Vector Double   -- ^ Coefficient estimates
                   -> Matrix Double    -- ^ Variance-covariance matrix
                   -> Double           -- ^ Critical value (e.g., 1.96 for 95% CI)
                   -> [(Double, Double)]  -- ^ List of (lower, upper) confidence bounds
confidenceIntervals beta vcov z =
    -- Validate dimensions
    if LA.size beta /= LA.rows vcov || LA.rows vcov /= LA.cols vcov
    then error $ "confidenceIntervals: Dimension mismatch between coefficients and variance matrix"
    else
        -- Compute standard errors
        let se = standardErrors vcov
        
        -- Calculate confidence intervals
        let lowerBounds = [b - z * e | (b, e) <- zip (LA.toList beta) (LA.toList se)]
            upperBounds = [b + z * e | (b, e) <- zip (LA.toList beta) (LA.toList se)]
        in zip lowerBounds upperBounds

-- | A record containing diagnostic information for a fitted GLM.
data DiagnosticReport = DiagnosticReport
    { fisherInfo :: Matrix Double         -- ^ Fisher Information Matrix
    , varianceCovMatrix :: Matrix Double  -- ^ Variance-covariance matrix (inverse of Fisher info)
    , stdErrors :: Vector Double          -- ^ Standard errors of coefficient estimates
    , confIntervals :: [(Double, Double)] -- ^ Confidence intervals for coefficients
    , coefs :: Vector Double              -- ^ Estimated coefficients
    , confLevel :: Double                 -- ^ Confidence level used
    }

instance Show DiagnosticReport where
    show report = 
        "GLM Diagnostic Report:\n" ++
        "---------------------\n" ++
        "Confidence Level: " ++ show (confLevel report * 100) ++ "%\n" ++
        formatCoefficients (coefs report) (stdErrors report) (confIntervals report)
      where
        formatCoefficients beta se ci =
            let rows = zip3 (LA.toList beta) (LA.toList se) ci
                header = "Coefficient | Std. Error | [" ++ show (confLevel report * 100) ++ "% Conf. Interval]\n"
                formatRow (i, (b, s, (l, u))) = 
                    printf "β_%d = %.6f | %.6f | [%.6f, %.6f]\n" i b s l u
                body = concatMap formatRow (zip [0..] rows)
            in header ++ body

-- | Generate a diagnostic report for a fitted GLM.
-- This computes all relevant diagnostic statistics and packages them into a report.
diagnosticReport :: Matrix Double   -- ^ Design matrix X
                -> Vector Double    -- ^ Working weights
                -> Vector Double    -- ^ Coefficient estimates
                -> Double           -- ^ Confidence level (e.g., 0.95 for 95% CI)
                -> DiagnosticReport -- ^ Diagnostic report
diagnosticReport x w beta confLevel =
    -- Compute Fisher Information Matrix
    let fi = fisherInformation x w
    
        -- Compute variance-covariance matrix
        vcov = varianceCovarianceMatrix fi
        
        -- Compute standard errors
        se = standardErrors vcov
        
        -- Compute confidence intervals
        z = criticalValue confLevel
        ci = confidenceIntervals beta vcov z
    
    in DiagnosticReport
        { fisherInfo = fi
        , varianceCovMatrix = vcov
        , stdErrors = se
        , confIntervals = ci
        , coefs = beta
        , confLevel = confLevel
        }
