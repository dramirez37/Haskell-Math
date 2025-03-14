# Haskell-Math

A collection of mathematical and statistical algorithms implemented in Haskell.

## Generalized Linear Models (GLM)

This library provides functionality for fitting Generalized Linear Models (GLMs) with optional LASSO regularization. The implementation offers multiple optimization algorithms:

1. Iteratively Reweighted Least Squares (IRLS) algorithm for parameter estimation
2. Pure coordinate descent solver for LASSO-regularized problems
3. Auto-selection of optimization method based on problem characteristics

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
- Multiple optimization algorithms:
  - IRLS with coordinate descent for LASSO
  - Pure coordinate descent solver
  - Factory pattern for pluggable optimizer architecture
- Detailed convergence tracking
- Econometric calculations for statistical inference:
  - Fisher Information Matrix computation
  - Coefficient variance-covariance estimation
  - Standard errors and confidence intervals
  - Comprehensive diagnostic reporting
- Safe matrix operations with HMatrix:
  - Dimension-checked matrix multiplication
  - Robust matrix inversion with pseudo-inverse fallback
  - Common matrix utilities with error handling
- Comprehensive error handling system:
  - Type-safe error propagation with Either monad
  - Detailed error messages for numerical issues
  - Integration with exception system when needed
  - Categories of errors for different problem types

### Error Handling System

The library implements a robust error handling system to provide clear, informative feedback about common numerical issues:

- **Typed errors**: Uses `GLMError` data type to categorize different error scenarios
- **Safe functions**: Key numerical operations return `Either GLMError a` instead of throwing exceptions
- **Detailed messages**: Context-specific error messages with relevant details (matrix dimensions, condition numbers)
- **Error propagation**: Consistent propagation through the optimizer architecture

Error categories include:
- Dimension mismatches in matrix operations
- Singular or ill-conditioned matrices
- Convergence failures
- Invalid input arguments
- General numerical issues

Example handling matrix operations with error checking:

```haskell
import Statistics.GLM.MatrixOperations
import Statistics.GLM.ErrorHandling
import Numeric.LinearAlgebra

main :: IO ()
main = do
    let a = matrix 3 3 [1,2,3,4,5,6,7,8,9]  -- A singular matrix
    
    -- Try to invert with proper error handling
    case safeInverse a of
        Right invA -> putStrLn "Success! Inverse:" >> print invA
        Left err -> putStrLn $ "Error: " ++ formatError err
        
    -- Try matrix multiplication with dimension checking
    let b = matrix 2 3 [1,2,3,4,5,6]
        c = matrix 2 2 [1,2,3,4]
    
    case safeMatrixMultiply b c of
        Right result -> putStrLn "Matrix product:" >> print result
        Left err -> putStrLn $ "Error: " ++ formatError err
```

### Modular Optimizer Architecture

The library utilizes a flexible optimizer architecture that allows you to:
- Select from different optimization algorithms
- Define custom optimizers by implementing the `Optimizer` interface
- Automatically choose the most appropriate method for a given problem

Example using the optimizer factory:

```haskell
import Statistics.GLM.Optimizer
import Statistics.GLM.ErrorHandling
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Create a model
    let x = matrix 10 3 $ \(i,j) -> if j == 0 then 1 else fromIntegral (i * j)
        y = vector [0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  -- Binary response
        model = modelFromMatrices x y bernoulliFamily logitLink Nothing
    
    -- Create configuration with LASSO regularization
    let config = defaultOptimizerConfig { lambdaLasso = 0.1, verbose = True }
    
    -- Create optimizer using factory (auto-selects appropriate method)
    let optimizer = createOptimizer AutoSelect config model
    
    -- Run optimization with error handling
    result <- optimize optimizer model config
    case result of
        Right optResult -> do
            putStrLn $ "Converged: " ++ show (converged optResult)
            putStrLn $ "Coefficients: " ++ show (coefficients optResult)
        Left err -> putStrLn $ "Error: " ++ formatError err
```

### Example Usage

```haskell
import Statistics.GLM.ExponentialFamily
import Statistics.GLM.IRLS
import Statistics.GLM.EconometricCalculations
import Statistics.GLM.MatrixOperations
import Statistics.GLM.ErrorHandling
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Create design matrix and response vector
    let x = matrix 10 3 $ \(i,j) -> if j == 0 then 1 else fromIntegral (i * j)
        y = vector [0, 0, 1, 0, 1, 1, 1, 1, 1, 1]  -- Binary response
    
    -- Fit a logistic regression model with proper error handling
    resultEither <- fitGLM x y bernoulliFamily logitLink defaultFitConfig
    
    case resultEither of
        Right result -> do
            -- Print the coefficients
            putStrLn "Estimated coefficients:"
            print $ coefficients result
            
            -- Generate diagnostic report with 95% confidence intervals
            let report = diagnosticReport 
                           (designMatrix result)
                           (workingWeights result)
                           (coefficients result)
                           0.95
            
            -- Print the diagnostic report
            putStrLn "\nStatistical inference:"
            print report
            
        Left err -> putStrLn $ "Error during model fitting: " ++ formatError err

    -- Example of safe matrix operations
    let a = matrix 2 2 [1,2,3,4]
        b = matrix 2 3 [5,6,7,8,9,10]
    
    -- Safe matrix multiplication with dimension checking
    case safeMatrixMultiply a b of
        Right c -> putStrLn "\nMatrix product:" >> print c
        Left err -> putStrLn $ "\nMultiplication error: " ++ formatError err
    
    -- Matrix inversion with error handling
    case safeInverse a of
        Right invA -> putStrLn "\nInverse matrix:" >> print invA
        Left err -> putStrLn $ "\nInversion error: " ++ formatError err
```

## Foreign Function Interface (FFI)

The library includes a Foreign Function Interface (FFI) that allows calling the GLM functionality from C, Python, and other languages.

### C-Callable Functions

The FFI module provides the following C-callable functions:

- `fitGLM_c`: Fit a GLM model with specified family, link function, and regularization
- `matrixMultiply_c`: Perform matrix multiplication with error handling
- `freeGLMResult` and `freeMatrix_c`: Free memory allocated by the library
- `getLastError_c`: Retrieve detailed error messages

### Example Usage from C

```c
#include <stdio.h>
#include <stdlib.h>

// Function prototypes matching the Haskell exports
extern int fitGLM_c(double* x, int nRows, int nCols, 
                    double* y, int ySize,
                    int family, int link, 
                    double lambda, int maxIter, double tol,
                    double** outCoefs, int* outSize);
                    
extern void freeGLMResult(double* coefficients);
extern char* getLastError_c(int* len);

int main() {
    // Create test data for logistic regression
    double x[] = {1.0, 0.5, 1.0, 1.2, 1.0, 2.3, 1.0, 3.1, 1.0, 4.0};
    double y[] = {0.0, 0.0, 1.0, 1.0, 1.0};
    
    // Output arrays
    double* coefficients = NULL;
    int coefSize = 0;
    
    // Call the GLM fitting function (logistic regression)
    int result = fitGLM_c(x, 5, 2,      // X matrix with 5 rows, 2 cols
                          y, 5,          // y vector with 5 elements
                          1, 1,          // Bernoulli family, logit link
                          0.0, 100, 1e-6, // No regularization, 100 max iter
                          &coefficients, &coefSize);
                          
    if (result == 0) {
        printf("Fit successful! Coefficients:\n");
        for (int i = 0; i < coefSize; i++) {
            printf("  beta_%d = %f\n", i, coefficients[i]);
        }
        
        // Free the allocated memory when done
        freeGLMResult(coefficients);
    } else {
        // Get error message
        int errLen = 0;
        char* errMsg = getLastError_c(&errLen);
        printf("Error %d: %s\n", result, errMsg);
        free(errMsg);  // Free the error string
    }
    
    return 0;
}
```

### Python Interface

The library provides a Python wrapper with two interfaces:

#### 1. Function-based API

Using the functional API for direct access to GLM capabilities:

```python
import numpy as np
from glm_wrapper import fit_glm, predict

# Create sample data (logistic regression example)
X = np.array([[1, 0.5], [1, 1.2], [1, 2.3], [1, 3.1], [1, 4.0]], dtype=np.float64)
y = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)

# Fit a logistic regression model
result = fit_glm(
    X, y,
    family='bernoulli',  # or use integer code 1
    link='logit',        # or use integer code 1
    lambda_val=0.1,      # L1 regularization parameter
    max_iter=100,
    tol=1e-6
)

if result['error'] == 0:
    print("Fit successful!")
    print("Coefficients:", result['coefficients'])
    
    # Make predictions on new data
    X_new = np.array([[1, 1.5], [1, 3.5]], dtype=np.float64)
    y_pred = predict(X_new, result['coefficients'], link='logit')
    print("Predictions:", y_pred)
else:
    print(f"Error {result['error']}: {result['error_message']}")
```

#### 2. Scikit-learn Compatible Class Interface

The library provides a comprehensive scikit-learn compatible `GLM` class that offers a familiar interface for Python users:

```python
import numpy as np
from glm_wrapper import GLM
from sklearn.model_selection import train_test_split

# Create sample data
X = np.array([[1, 0.5], [1, 1.2], [1, 2.3], [1, 3.1], [1, 4.0], 
              [1, 2.5], [1, 3.2], [1, 3.8], [1, 4.5], [1, 5.1]], dtype=np.float64)
y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.float64)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit a logistic regression model with LASSO
model = GLM(family='bernoulli', link='logit', lambda_val=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Predicted probabilities:", y_pred)

# Convert to binary predictions
y_pred_binary = model.predict(X_test, output_type='class')
print("Binary predictions:", y_pred_binary)

# Get model accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Access the coefficients
print("Model coefficients:", model.coef_)

# Get a detailed model summary
print(model.summary())
```

#### GLM Class Features

The `GLM` class provides the following capabilities:

- **Flexible Model Specification**: Supports all distribution families and link functions
- **Multiple Output Types**: Predict raw values, linear predictor values, or classes
- **Model Evaluation**: Built-in scoring functions appropriate to each distribution family
- **Model Summaries**: Generate comprehensive model reports with `summary()`
- **Scikit-learn Integration**: Compatible with scikit-learn's API standards including:
  - Method chaining with `fit()` returning `self`
  - Standard attribute naming with `coef_`, `n_iter_`, etc.
  - Parameter management with `get_params()` and `set_params()`
  - Consistent error handling with informative messages

#### Distribution Families and Link Functions

| Family     | Description                     | Canonical Link | Typical Use Case     |
|------------|---------------------------------|---------------|----------------------|
| gaussian   | Normal distribution             | identity      | Continuous responses |
| bernoulli  | Binary distribution (0/1)       | logit         | Binary classification|
| poisson    | Count distribution              | log           | Count data           |

#### Exception Handling

The Python API includes a comprehensive exception handling system that converts Haskell-side error codes into Python exceptions:

| Exception Type | Description | Common Causes |
|----------------|-------------|---------------|
| `GLMError` | Base class for all exceptions | General or unknown errors |
| `DimensionMismatchError` | Matrix dimension incompatibility | Incorrect input shapes |
| `SingularMatrixError` | Matrix is singular | Linearly dependent columns |
| `IllConditionedError` | Matrix is poorly conditioned | Near multicollinearity |
| `NumericError` | Numerical computation failure | Overflow, underflow, NaN |
| `InvalidArgumentError` | Invalid function arguments | Wrong family/link combination |
| `ConvergenceError` | Algorithm failed to converge | Insufficient iterations |
| `ImplementationError` | Internal library bug | Implementation defects |

**Example: Handling Specific Exceptions**

```python
from glm_wrapper import GLM
from glm_wrapper.Exceptions import DimensionMismatchError, ConvergenceError, GLMError

try:
    # Try to fit a GLM model
    model = GLM(family='bernoulli', link='logit')
    model.fit(X, y)
    predictions = model.predict(X_new)
    
except DimensionMismatchError as e:
    print(f"Input dimension problem: {e}")
    print("Check that X and y have compatible shapes")
    
except ConvergenceError as e:
    print(f"Model failed to converge: {e}")
    print("Try increasing max_iter or adjusting lambda_val")
    
except GLMError as e:
    print(f"Other GLM error occurred: {e}")
```

**Integration with Scientific Python Ecosystem**

These exceptions are designed to integrate well with NumPy and other scientific Python libraries. They provide detailed error messages and include the original Haskell error code for debugging:

```python
try:
    result = fit_glm(X, y, family='bernoulli')
except GLMError as e:
    print(f"Error code: {e.error_code}")
    print(f"Error message: {str(e)}")
```

When using the scikit-learn compatible interface, errors are automatically converted to Python exceptions, allowing for clean error handling in data science workflows including those that use pipelines and grid search.

### Error Handling in FFI

The FFI module uses numeric error codes to communicate errors to C/Python:

| Error Code | Description |
|------------|-------------|
| 0 | Success |
| -1 | Dimension mismatch |
| -2 | Singular matrix |
| -3 | Ill-conditioned matrix |
| -4 | General numeric error |
| -5 | Invalid argument |
| -6 | Convergence failure |
| -7 | Implementation error |
| -99 | Unknown error |

Detailed error messages can be retrieved using the `getLastError_c` function.

### Memory Management

When using the FFI functions that return allocated memory (like `fitGLM_c` or `matrixMultiply_c`), 
always call the corresponding free function (like `freeGLMResult` or `freeMatrix_c`) when you're 
done with the result. This prevents memory leaks since memory allocated in Haskell must be freed 
by Haskell.

In the Python wrapper, memory management is handled automatically, so you don't need to worry about 
calling the free functions directly.

### Testing

The library includes a comprehensive testing infrastructure to validate the correctness and performance of the implementation.

#### Integration Tests

Integration tests compare the GLM implementation against established libraries like scikit-learn and statsmodels to ensure consistent results. 

These tests:
- Generate synthetic data for Gaussian, Bernoulli, and Poisson distributions
- Fit models using both our GLM implementation and reference libraries
- Compare coefficient estimates, predictions, and performance metrics
- Report detailed comparisons and execution times

To run the integration tests:

```bash
python tests/Integration_Tests.py
```

Sample output:

## License

This project is licensed under the Apache License, Version 2.0.