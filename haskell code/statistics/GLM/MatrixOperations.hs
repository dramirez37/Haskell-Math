{-|
Module      : Statistics.GLM.MatrixOperations
Description : Safe wrappers for matrix operations using HMatrix
Copyright   : (c) 2025
License     : Apache 2.0

This module provides utility functions that wrap common matrix operations from HMatrix,
adding dimension checks, error handling, and other safety features.
-}

module Statistics.GLM.MatrixOperations 
(
  -- * Safe matrix operations with dimension checks
  safeMultiply
, safeMatrixMultiply
, safeInverse
, matrixTranspose
  -- * Additional helper functions
, scaleMatrix
, elementWiseOp
, diagonalMatrix
, isDiagonallyDominant
) where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, Vector, (<>), (<.>))
import Text.Printf (printf)
import Statistics.GLM.ErrorHandling

-- | Safely multiply two matrices with dimension checking.
-- Returns the product if dimensions match, otherwise crashes with an error message.
safeMultiply :: Matrix Double  -- ^ First matrix (m × n)
             -> Matrix Double  -- ^ Second matrix (n × p)
             -> Matrix Double  -- ^ Result matrix (m × p)
safeMultiply m1 m2 =
    let (m1rows, m1cols) = LA.size m1
        (m2rows, m2cols) = LA.size m2
    in
        if m1cols == m2rows
        then m1 LA.<> m2
        else error $ formatError $ DimensionMismatchError $
             dimensionErrorMsg "Matrix multiplication" 
                            [m1rows, m1cols] [m2rows, m2cols]

-- | Safely multiply two matrices with dimension checking.
-- Returns Either an error message or the product.
safeMatrixMultiply :: Matrix Double  -- ^ First matrix (m × n)
                   -> Matrix Double  -- ^ Second matrix (n × p)
                   -> Either GLMError (Matrix Double)  -- ^ Either error message or result matrix (m × p)
safeMatrixMultiply m1 m2 =
    let (m1rows, m1cols) = LA.size m1
        (m2rows, m2cols) = LA.size m2
    in
        if m1cols == m2rows
        then Right (m1 LA.<> m2)
        else Left $ DimensionMismatchError $
             dimensionErrorMsg "Matrix multiplication" 
                            [m1rows, m1cols] [m2rows, m2cols]

-- | Safely invert a matrix.
-- Returns Either an error message or the inverse matrix.
-- If the matrix is singular or nearly singular, we attempt to compute a pseudo-inverse.
safeInverse :: Matrix Double  -- ^ Matrix to invert
           -> Either GLMError (Matrix Double)  -- ^ Either error message or inverse matrix
safeInverse mat =
    let (rows, cols) = LA.size mat
    in
        if rows /= cols
        then Left $ DimensionMismatchError $
             printf "Cannot invert non-square matrix of size %d×%d" rows cols
        else 
            -- Check condition number as a measure of numerical stability
            let (_, s, _) = LA.svd mat
                condNumber = maximum (LA.toList s) / minimum (LA.toList s)
            in
                if condNumber > 1e15  -- Extremely ill-conditioned
                then Left $ IllConditionedError $
                     conditionNumberErrorMsg "Matrix inversion" condNumber
                else 
                    -- Try standard inversion first
                    if LA.rank mat == rows
                    then Right $ LA.inv mat  -- Standard inversion for full rank matrices
                    else 
                        -- Matrix is singular, return pseudo-inverse with warning
                        Left $ SingularMatrixError $
                        rankDeficientErrorMsg "Matrix inversion" 
                                           (LA.rank mat) rows

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

-- | Transpose a matrix.
-- This is a simple wrapper around HMatrix's transpose function.
matrixTranspose :: Matrix Double  -- ^ Matrix to transpose
               -> Matrix Double   -- ^ Transposed matrix
matrixTranspose = LA.tr

-- | Scale a matrix by a scalar value.
scaleMatrix :: Double          -- ^ Scalar multiplier
           -> Matrix Double    -- ^ Input matrix
           -> Matrix Double    -- ^ Scaled matrix
scaleMatrix scalar = LA.scale scalar

-- | Perform element-wise operation on two matrices.
-- Matrices must have the same dimensions.
elementWiseOp :: (Double -> Double -> Double)  -- ^ Binary operation to apply
             -> Matrix Double                 -- ^ First matrix
             -> Matrix Double                 -- ^ Second matrix
             -> Either GLMError (Matrix Double)  -- ^ Either error message or result matrix
elementWiseOp op m1 m2 =
    let (m1rows, m1cols) = LA.size m1
        (m2rows, m2cols) = LA.size m2
    in
        if m1rows == m2rows && m1cols == m2cols
        then Right $ LA.fromList m1rows m1cols $
             zipWith op (LA.toList (LA.flatten m1)) (LA.toList (LA.flatten m2)) 
        else Left $ DimensionMismatchError $
             dimensionErrorMsg "Element-wise operation" 
                            [m1rows, m1cols] [m2rows, m2cols]

-- | Create a diagonal matrix from a vector.
diagonalMatrix :: Vector Double  -- ^ Diagonal elements
              -> Matrix Double   -- ^ Resulting diagonal matrix
diagonalMatrix = LA.diag

-- | Check if a matrix is diagonally dominant.
-- A matrix is diagonally dominant if for each row,
-- the absolute value of the diagonal entry is greater
-- than or equal to the sum of absolute values of other entries.
isDiagonallyDominant :: Matrix Double  -- ^ Matrix to check
                    -> Bool           -- ^ Whether the matrix is diagonally dominant
isDiagonallyDominant mat =
    let (rows, cols) = LA.size mat
    in
        if rows /= cols
        then False  -- Non-square matrices are not diagonally dominant
        else
            let 
                -- Check each row for diagonal dominance
                checkRow i =
                    let diag = abs (mat `LA.atIndex` (i,i))
                        rowSum = sum [abs (mat `LA.atIndex` (i,j)) | j <- [0..cols-1], j /= i]
                    in diag >= rowSum
            in
                all checkRow [0..rows-1]
