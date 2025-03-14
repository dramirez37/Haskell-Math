{-|
Module      : Statistics.GLM.ErrorHandling
Description : Error handling system for GLM library
Copyright   : (c) 2025
License     : Apache 2.0

This module provides a comprehensive error handling system for the GLM library,
defining custom error types and formatting functions for common numerical issues.
-}

module Statistics.GLM.ErrorHandling 
(
  -- * Error Types
  GLMError(..),
  
  -- * Error Formatting
  formatError,
  
  -- * Helper Functions
  dimensionErrorMsg,
  conditionNumberErrorMsg,
  rankDeficientErrorMsg,
  
  -- * Type-safe error propagation
  mapLeft,
  propagateError,
  
  -- * Error throwing/catching utilities
  fromEither,
  toException,
  fromException,
  glmToException,
  glmFromException
) where

import Control.Exception (Exception(..), SomeException)
import qualified Control.Exception as E
import Data.Typeable (Typeable, cast)
import Text.Printf (printf)
import Numeric.LinearAlgebra (Matrix, Vector, rows, cols, size)
import qualified Numeric.LinearAlgebra as LA

-- | Custom error type for GLM operations
data GLMError
  = DimensionMismatchError String   -- ^ For incompatible matrix dimensions
  | SingularMatrixError String      -- ^ For singular or nearly singular matrices
  | IllConditionedError String      -- ^ For matrices with high condition numbers
  | GeneralNumericError String      -- ^ For other numerical issues
  | InvalidArgumentError String     -- ^ For invalid function arguments
  | ConvergenceError String         -- ^ For algorithms that fail to converge
  | ImplementationError String      -- ^ For internal implementation errors
  deriving (Eq, Typeable)

instance Show GLMError where
  show = formatError

-- Make GLMError an instance of Exception
instance Exception GLMError where
  toException = E.toException
  fromException = E.fromException

-- | Convert GLMError to a human-readable message
formatError :: GLMError -> String
formatError (DimensionMismatchError details) =
  "Dimension Mismatch Error: " ++ details
formatError (SingularMatrixError details) =
  "Singular Matrix Error: " ++ details
formatError (IllConditionedError details) =
  "Ill-Conditioned Matrix Error: " ++ details
formatError (GeneralNumericError details) =
  "Numerical Error: " ++ details
formatError (InvalidArgumentError details) =
  "Invalid Argument Error: " ++ details
formatError (ConvergenceError details) =
  "Convergence Error: " ++ details
formatError (ImplementationError details) =
  "Implementation Error: " ++ details

-- | Helper function to format dimension mismatch errors
dimensionErrorMsg :: String      -- ^ Operation name
                  -> [Int]       -- ^ First dimensions
                  -> [Int]       -- ^ Second dimensions
                  -> String
dimensionErrorMsg op dims1 dims2 =
  printf "%s operation failed: incompatible dimensions %s and %s"
         op (show dims1) (show dims2)

-- | Helper function to format condition number errors
conditionNumberErrorMsg :: String  -- ^ Operation name
                        -> Double  -- ^ Condition number
                        -> String
conditionNumberErrorMsg op condNum =
  printf "%s failed: matrix is ill-conditioned with condition number %.2e (> 1e15)"
         op condNum

-- | Helper function to format rank deficient matrix errors
rankDeficientErrorMsg :: String  -- ^ Operation name
                      -> Int     -- ^ Actual rank
                      -> Int     -- ^ Expected rank
                      -> String
rankDeficientErrorMsg op actualRank expectedRank =
  printf "%s failed: matrix is rank deficient (rank %d, expected %d)"
         op actualRank expectedRank

-- | Helper function to map over the Left part of an Either
mapLeft :: (a -> b) -> Either a c -> Either b c
mapLeft f (Left x) = Left (f x)
mapLeft _ (Right x) = Right x

-- | Helper function to propagate errors through function composition
propagateError :: Either GLMError a -> (a -> Either GLMError b) -> Either GLMError b
propagateError (Left err) _ = Left err
propagateError (Right x) f = f x

-- | Convert Either to a value, throwing an exception on Left
fromEither :: Either GLMError a -> IO a
fromEither (Right x) = return x
fromEither (Left err) = E.throwIO err

-- | Convert GLMError to SomeException
toException :: GLMError -> SomeException
toException = E.toException

-- | Convert SomeException to Maybe GLMError
fromException :: SomeException -> Maybe GLMError
fromException = E.fromException

-- | Helper function to check if a matrix is approximately singular
isApproxSingular :: Matrix Double -> Bool
isApproxSingular m = 
  let (_, s, _) = LA.svd m
      minSingularValue = minimum (LA.toList s)
  in minSingularValue < 1e-10

-- | Helper function to compute condition number of a matrix
conditionNumber :: Matrix Double -> Double
conditionNumber m =
  let (_, s, _) = LA.svd m
      maxS = maximum (LA.toList s)
      minS = maximum [minimum (LA.toList s), 1e-15]  -- Avoid division by zero
  in maxS / minS

-- | Check if dimensions are compatible for matrix multiplication
checkMultiplyDimensions :: Matrix Double -> Matrix Double -> Either GLMError ()
checkMultiplyDimensions m1 m2 =
  let (r1, c1) = (rows m1, cols m1)
      (r2, c2) = (rows m2, cols m2)
  in if c1 == r2
     then Right ()
     else Left $ DimensionMismatchError $
          dimensionErrorMsg "Matrix multiplication" [r1, c1] [r2, c2]

-- | Convert SomeException to SomeException
glmToException :: GLMError -> SomeException
glmToException = E.toException

-- | Convert SomeException to Maybe GLMError
glmFromException :: SomeException -> Maybe GLMError
glmFromException = cast
