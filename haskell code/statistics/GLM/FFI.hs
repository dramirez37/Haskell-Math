{-|
Module      : Statistics.GLM.FFI
Description : Foreign Function Interface for the GLM library
Copyright   : (c) 2025
License     : Apache 2.0

This module provides C-callable functions for the GLM library,
enabling integration with other languages like Python via ctypes or cffi.
-}

module Statistics.GLM.FFI 
(
  -- * C-callable functions
  fitGLM_c
, freeGLMResult
, getLastError_c

  -- * Error codes
, c_SUCCESS
, c_DIMENSION_MISMATCH
, c_SINGULAR_MATRIX
, c_ILL_CONDITIONED
, c_NUMERIC_ERROR
, c_INVALID_ARGUMENT
, c_CONVERGENCE_ERROR
, c_IMPLEMENTATION_ERROR
, c_UNKNOWN_ERROR
) where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.StablePtr
import Control.Exception (catch, SomeException)
import System.IO.Unsafe (unsafePerformIO)

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, Vector, (<>), (<.>))

import Statistics.GLM.ExponentialFamily
import Statistics.GLM.IRLS
import Statistics.GLM.ErrorHandling
import Statistics.GLM.MatrixOperations

-- | Global variable to store the last error message

-- | Error codes for C/Python clients
c_SUCCESS, c_DIMENSION_MISMATCH, c_SINGULAR_MATRIX, c_ILL_CONDITIONED, 
  c_NUMERIC_ERROR, c_INVALID_ARGUMENT, c_CONVERGENCE_ERROR, 
  c_IMPLEMENTATION_ERROR, c_UNKNOWN_ERROR :: CInt

c_SUCCESS = 0
c_DIMENSION_MISMATCH = -1
c_SINGULAR_MATRIX = -2
c_ILL_CONDITIONED = -3
c_NUMERIC_ERROR = -4
c_INVALID_ARGUMENT = -5
c_CONVERGENCE_ERROR = -6
c_IMPLEMENTATION_ERROR = -7
c_UNKNOWN_ERROR = -99

-- | Convert GLMError to C error code
glmErrorToCode :: GLMError -> CInt
glmErrorToCode (DimensionMismatchError _) = c_DIMENSION_MISMATCH
glmErrorToCode (SingularMatrixError _) = c_SINGULAR_MATRIX
glmErrorToCode (IllConditionedError _) = c_ILL_CONDITIONED
glmErrorToCode (GeneralNumericError _) = c_NUMERIC_ERROR
glmErrorToCode (InvalidArgumentError _) = c_INVALID_ARGUMENT
glmErrorToCode (ConvergenceError _) = c_CONVERGENCE_ERROR
glmErrorToCode (ImplementationError _) = c_IMPLEMENTATION_ERROR

-- | Store error message and return error code
setError :: GLMError -> IO CInt
setError err = do
    writeIORef lastErrorMsg (formatError err)
    return (glmErrorToCode err)

-- | Handle exceptions and convert to error codes
handleException :: SomeException -> IO CInt
handleException ex = do
    let msg = "Unexpected error: " ++ show ex
    writeIORef lastErrorMsg msg
    return c_UNKNOWN_ERROR

-- | Convert raw C arrays to HMatrix types
createMatrixFromCArray :: Ptr CDouble -> CInt -> CInt -> IO (Matrix Double)
createMatrixFromCArray ptr rows cols = do
    let n = fromIntegral (rows * cols)
    arr <- peekArray n ptr
    return $ LA.reshape (fromIntegral cols) $ LA.vector $ map realToFrac arr

createVectorFromCArray :: Ptr CDouble -> CInt -> IO (Vector Double)
createVectorFromCArray ptr size = do
    arr <- peekArray (fromIntegral size) ptr
    return $ LA.vector $ map realToFrac arr

-- | Determine family type from integer code
getFamilyFromInt :: CInt -> Either GLMError ExponentialFamily
getFamilyFromInt 0 = Right gaussianFamily
getFamilyFromInt 1 = Right bernoulliFamily
getFamilyFromInt 2 = Right poissonFamily
getFamilyFromInt n = Left $ InvalidArgumentError $ 
    "Invalid family code: " ++ show n ++ ". Use 0=Gaussian, 1=Bernoulli, 2=Poisson"

-- | Determine link function from integer code
getLinkFromInt :: CInt -> Either GLMError Link
getLinkFromInt 0 = Right identityLink
getLinkFromInt 1 = Right logitLink
getLinkFromInt 2 = Right logLink
getLinkFromInt n = Left $ InvalidArgumentError $ 
    "Invalid link code: " ++ show n ++ ". Use 0=identity, 1=logit, 2=log"

-- | Copy a Vector to newly allocated memory for returning to C
vectorToCArray :: Vector Double -> IO (Ptr CDouble, CInt)
vectorToCArray vec = do
    let n = LA.size vec
    ptr <- mallocArray n
    pokeArray ptr (map realToFrac $ LA.toList vec)
    return (ptr, fromIntegral n)

-- | Foreign exported function to fit a GLM model
foreign export ccall fitGLM_c :: Ptr CDouble -> CInt -> CInt -> 
                               Ptr CDouble -> CInt -> 
                               CInt -> CInt -> 
                               CDouble -> CInt -> CDouble ->
                               Ptr (Ptr CDouble) -> Ptr CInt ->
                               IO CInt

fitGLM_c :: Ptr CDouble   -- ^ Design matrix X (flattened in row-major order)
         -> CInt          -- ^ Number of rows in X
         -> CInt          -- ^ Number of columns in X
         -> Ptr CDouble   -- ^ Response vector y
         -> CInt          -- ^ Size of y
         -> CInt          -- ^ Family code (0=Gaussian, 1=Bernoulli, 2=Poisson)
         -> CInt          -- ^ Link code (0=identity, 1=logit, 2=log)
         -> CDouble       -- ^ LASSO regularization parameter lambda
         -> CInt          -- ^ Maximum iterations
         -> CDouble       -- ^ Convergence tolerance
         -> Ptr (Ptr CDouble) -- ^ Pointer to receive the coefficients array
         -> Ptr CInt      -- ^ Pointer to receive the size of coefficients array
         -> IO CInt       -- ^ Error code
fitGLM_c xPtr nRows nCols yPtr ySize famCode linkCode lambda maxIter tol outCoefs outSize =
    (do
        -- Convert C arrays to HMatrix types
        x <- createMatrixFromCArray xPtr nRows nCols
        y <- createVectorFromCArray yPtr ySize
        
        -- Check dimensions
        if LA.rows x /= LA.size y
            then setError $ DimensionMismatchError $ 
                 dimensionErrorMsg "GLM fitting" [fromIntegral nRows, fromIntegral nCols] 
                                 [fromIntegral ySize, 1]
            else do
                -- Get family and link
                case getFamilyFromInt famCode of
                    Left err -> setError err
                    Right family -> case getLinkFromInt linkCode of
                        Left err -> setError err
                        Right link -> do
                            -- Create fit configuration
                            let config = FitConfig {
                                maxIterations = fromIntegral maxIter,
                                tolerance = realToFrac tol,
                                llTolerance = 1e-8,
                                lambdaLasso = realToFrac lambda,
                                verbose = False,
                                addIntercept = True
                            }
                            
                            -- Fit the model
                            result <- fitGLM x y family link config
                            
                            case result of
                                Left err -> setError err
                                Right fitResult -> do
                                    -- Get coefficients and copy to C array
                                    let coefs = coefficients fitResult
                                    (coefsPtr, coefsSize) <- vectorToCArray coefs
                                    
                                    -- Set output parameters
                                    poke outCoefs coefsPtr
                                    poke outSize coefsSize
                                    
                                    -- Return success
                                    return c_SUCCESS
    ) `catch` handleException

-- | Foreign exported function to free memory allocated by fitGLM_c
foreign export ccall freeGLMResult :: Ptr CDouble -> IO ()

freeGLMResult :: Ptr CDouble -> IO ()
freeGLMResult ptr = free ptr

-- | Foreign exported function to get the last error message
foreign export ccall getLastError_c :: Ptr CInt -> IO CString

getLastError_c :: Ptr CInt -> IO CString
getLastError_c lenPtr = do
    msg <- readIORef lastErrorMsg
    poke lenPtr (fromIntegral $ length msg)
    newCString msg

-- | Helper function to test matrix operations through FFI
foreign export ccall matrixMultiply_c :: Ptr CDouble -> CInt -> CInt ->
                                        Ptr CDouble -> CInt -> CInt ->
                                        Ptr (Ptr CDouble) -> Ptr CInt -> Ptr CInt ->
                                        IO CInt

matrixMultiply_c :: Ptr CDouble -> CInt -> CInt -> 
                   Ptr CDouble -> CInt -> CInt ->
                   Ptr (Ptr CDouble) -> Ptr CInt -> Ptr CInt ->
                   IO CInt
matrixMultiply_c m1Ptr m1Rows m1Cols m2Ptr m2Rows m2Cols outPtr outRows outCols =
    (do
        -- Convert C arrays to matrices
        m1 <- createMatrixFromCArray m1Ptr m1Rows m1Cols
        m2 <- createMatrixFromCArray m2Ptr m2Rows m2Cols
        
        -- Multiply matrices with error handling
        case safeMatrixMultiply m1 m2 of
            Left err -> setError err
            Right result -> do
                -- Get dimensions of result
                let rows = LA.rows result
                    cols = LA.cols result
                    flatSize = rows * cols
                
                -- Allocate memory for result
                resultPtr <- mallocArray flatSize
                
                -- Copy data to result array
                let flattenedData = map realToFrac $ LA.toList $ LA.flatten result
                pokeArray resultPtr flattenedData
                
                -- Set output parameters
                poke outPtr resultPtr
                poke outRows (fromIntegral rows)
                poke outCols (fromIntegral cols)
                
                return c_SUCCESS
    ) `catch` handleException

-- | Foreign exported function to free a matrix result
foreign export ccall freeMatrix_c :: Ptr CDouble -> IO ()

freeMatrix_c :: Ptr CDouble -> IO ()
freeMatrix_c = free
