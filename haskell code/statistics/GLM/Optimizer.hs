{-|
Module      : Statistics.GLM.Optimizer
Description : Modular optimizer architecture with factory pattern for GLMs
Copyright   : (c) 2025
License     : Apache 2.0

This module provides a flexible optimizer architecture that allows plugging in
different optimization algorithms for Generalized Linear Models (GLMs).
-}

module Statistics.GLM.Optimizer 
(
  -- * Core interfaces and types
  Optimizer(..)
, Model(..)
, OptimizerConfig(..)
, OptimizerType(..)
, OptResult(..)
, SomeOptimizer(..)
  
  -- * Factory functions
, createOptimizer
, defaultOptimizerConfig
, modelFromMatrices

  -- * Specific optimizer implementations (typically not used directly)
, IRLSOptimizer(..)
, CoordinateDescentOptimizer(..)
) where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, Vector, (<>), (<.>))
import Statistics.GLM.ExponentialFamily
import qualified Statistics.GLM.IRLS as IRLS
import qualified Statistics.GLM.CoordinateDescent as CD
import qualified Statistics.GLM.MatrixOperations as MO
import Statistics.GLM.ErrorHandling
import Data.Maybe (fromMaybe)

-- | The core model data required for GLM optimization
data Model = Model
  { designMatrix :: Matrix Double       -- ^ Design matrix X
  , responseVector :: Vector Double     -- ^ Response vector y
  , family :: ExponentialFamily         -- ^ Exponential family distribution
  , link :: Link                        -- ^ Link function
  , weights :: Maybe (Vector Double)    -- ^ Optional observation weights
  } deriving Show

-- | Common configuration for all optimizers
data OptimizerConfig = OptimizerConfig
  { maxIterations :: Int                -- ^ Maximum number of iterations
  , tolerance :: Double                 -- ^ Convergence tolerance for coefficient changes
  , llTolerance :: Maybe Double         -- ^ Log-likelihood convergence tolerance
  , lambdaLasso :: Double               -- ^ LASSO regularization parameter (0 = no regularization)
  , verbose :: Bool                     -- ^ Whether to print progress information
  , addIntercept :: Bool                -- ^ Whether to add an intercept term automatically
  } deriving Show

-- | Default optimizer configuration
defaultOptimizerConfig :: OptimizerConfig
defaultOptimizerConfig = OptimizerConfig
  { maxIterations = 100
  , tolerance = 1e-6
  , llTolerance = Just 1e-8
  , lambdaLasso = 0.0
  , verbose = False
  , addIntercept = True
  }

-- | Available optimizer types
data OptimizerType = IRLS | CoordinateDescent | AutoSelect
  deriving (Show, Eq)

-- | Result of optimization process
data OptResult = OptResult
  { coefficients :: Vector Double       -- ^ Estimated coefficients
  , iterations :: Int                   -- ^ Number of iterations performed
  , converged :: Bool                   -- ^ Whether the algorithm converged
  , finalLogLikelihood :: Maybe Double  -- ^ Final log-likelihood value if available
  , message :: String                   -- ^ Message about convergence status
  , workingWeights :: Maybe (Vector Double) -- ^ Final working weights (for inference)
  } deriving Show

-- | Core optimizer interface that all optimizers must implement
class Optimizer opt where
  optimize :: opt -> Model -> OptimizerConfig -> IO (Either GLMError OptResult)
  
  -- | Optional method to check if optimizer is appropriate for given model/config
  isAppropriate :: opt -> Model -> OptimizerConfig -> Bool
  isAppropriate _ _ _ = True  -- Default implementation: any optimizer works for any model

-- | Existential wrapper to hide concrete optimizer types
data SomeOptimizer = forall opt. Optimizer opt => SomeOptimizer opt

-- Make SomeOptimizer an instance of Optimizer to use it polymorphically
instance Optimizer SomeOptimizer where
  optimize (SomeOptimizer opt) model config = optimize opt model config
  isAppropriate (SomeOptimizer opt) model config = isAppropriate opt model config

-- | Helper function to construct a Model from matrices
modelFromMatrices :: Matrix Double -> Vector Double -> ExponentialFamily -> Link 
                  -> Maybe (Vector Double) -> Model
modelFromMatrices x y fam lnk wts = Model x y fam lnk wts

-- | IRLS optimizer implementation
data IRLSOptimizer = IRLSOptimizer

instance Optimizer IRLSOptimizer where
  optimize _ model config = do
    -- Convert our Model and OptimizerConfig to IRLS.FitConfig
    let irlsConfig = IRLS.FitConfig
          { IRLS.maxIterations = maxIterations config
          , IRLS.tolerance = tolerance config
          , IRLS.llTolerance = fromMaybe 1e-8 (llTolerance config)
          , IRLS.lambdaLasso = lambdaLasso config
          , IRLS.verbose = verbose config
          , IRLS.addIntercept = False  -- We handle this at the Model level
          }
    
    -- Apply intercept if requested
    let xMatrix = if addIntercept config && not (hasIntercept (designMatrix model))
                  then addInterceptColumn (designMatrix model)
                  else designMatrix model
    
    -- Call IRLS.fitGLM
    result <- IRLS.fitGLM 
                xMatrix
                (responseVector model)
                (family model)
                (link model)
                irlsConfig
    
    -- Convert IRLS.FitResult to our OptResult or propagate error
    case result of
      Right fitResult -> 
        return $ Right OptResult
          { coefficients = IRLS.coefficients fitResult
          , iterations = IRLS.iterations (IRLS.convergenceInfo fitResult)
          , converged = IRLS.converged (IRLS.convergenceInfo fitResult)
          , finalLogLikelihood = Just (IRLS.finalLogLikelihood (IRLS.convergenceInfo fitResult))
          , message = IRLS.message (IRLS.convergenceInfo fitResult)
          , workingWeights = Just (IRLS.workingWeights fitResult)
          }
      Left err -> return $ Left err
  
  -- IRLS is appropriate for all GLM problems but especially good for medium-sized problems
  isAppropriate _ model config = 
    let n = LA.rows (designMatrix model)
        p = LA.cols (designMatrix model)
    in n <= 100000 && p <= 1000  -- IRLS is efficient for moderate-sized problems

-- | Pure Coordinate Descent optimizer implementation
data CoordinateDescentOptimizer = CoordinateDescentOptimizer

instance Optimizer CoordinateDescentOptimizer where
  optimize _ model config = do
    -- Apply intercept if requested
    let xMatrix = if addIntercept config && not (hasIntercept (designMatrix model))
                  then addInterceptColumn (designMatrix model)
                  else designMatrix model
        
        x = xMatrix
        y = responseVector model
        fam = family model
        lnk = link model
        lambda = lambdaLasso config
        
        -- Initialize with zeros or use a smarter initialization
        beta0 = LA.konst 0 (LA.cols x)
        
        -- For coordinate descent, we need working weights and responses
        -- Here we compute initial values (a real implementation would be more sophisticated)
        workingWeights = LA.konst 1 (LA.rows x)  -- Simplified - actual weights depend on family/link
                
        -- Call the coordinate descent solver
        result = CD.coordinateDescentSolver 
                  x y workingWeights lambda beta0
                  (maxIterations config) (tolerance config) (verbose config)
    
    return $ Right OptResult
      { coefficients = CD.cdCoefficients result
      , iterations = CD.cdIterations (CD.cdConvergenceInfo result)
      , converged = CD.cdConverged (CD.cdConvergenceInfo result)
      , finalLogLikelihood = Nothing  -- CD doesn't track log-likelihood directly
      , message = CD.cdMessage (CD.cdConvergenceInfo result)
      , workingWeights = Just workingWeights
      }
  
  -- Coordinate Descent is especially appropriate for LASSO problems with many features
  isAppropriate _ _ config = lambdaLasso config > 0  -- CD is great for L1 regularization

-- | Factory function to create an optimizer based on type and config
createOptimizer :: OptimizerType -> OptimizerConfig -> Model -> SomeOptimizer
createOptimizer IRLS _ _ = SomeOptimizer IRLSOptimizer
createOptimizer CoordinateDescent _ _ = SomeOptimizer CoordinateDescentOptimizer
createOptimizer AutoSelect config model = 
    -- Auto-select the most appropriate optimizer based on problem characteristics
    if lambdaLasso config > 0 && LA.cols (designMatrix model) > 1000
    then SomeOptimizer CoordinateDescentOptimizer  -- For high-dimensional LASSO
    else SomeOptimizer IRLSOptimizer               -- Default to IRLS for most problems

-- | Helper function to check if matrix already has an intercept column
hasIntercept :: Matrix Double -> Bool
hasIntercept x = 
    let firstCol = LA.takeColumn x 0
        isConstant = all (\v -> abs (v - (firstCol LA.! 0)) < 1e-10) (LA.toList firstCol)
    in isConstant && LA.cols x > 0

-- | Helper function to add intercept column to design matrix
addInterceptColumn :: Matrix Double -> Matrix Double
addInterceptColumn x = 
    let n = LA.rows x
        intercept = LA.konst 1 n
    in LA.fromColumns (intercept : LA.toColumns x)
