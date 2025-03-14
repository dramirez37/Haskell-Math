import ctypes
import numpy as np
import os
import platform
from numpy.ctypeslib import ndpointer

# ===============================================================
# Library loading and error code definitions
# ===============================================================

def _find_lib():
    """Find the GLM shared library based on the platform."""
    system = platform.system()
    if system == "Linux":
        return "libglm.so"
    elif system == "Darwin":
        return "libglm.dylib"
    elif system == "Windows":
        return "glm.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

# Try to load the library from several common locations
def _load_lib():
    lib_name = _find_lib()
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),  # Same directory as this script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"),  # lib subdirectory
        "/usr/local/lib",  # Common location on Unix-like systems
        "/usr/lib",  # Another common location
    ]
    
    for path in search_paths:
        try:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)
        except OSError:
            continue
    
    # Last resort: try to load by name only
    try:
        return ctypes.CDLL(lib_name)
    except OSError:
        raise RuntimeError(f"Could not find {lib_name}. Make sure it's installed and in your library path.")

# Error codes
SUCCESS = 0
DIMENSION_MISMATCH = -1
SINGULAR_MATRIX = -2
ILL_CONDITIONED = -3
NUMERIC_ERROR = -4
INVALID_ARGUMENT = -5
CONVERGENCE_ERROR = -6
IMPLEMENTATION_ERROR = -7
UNKNOWN_ERROR = -99

# Error message dictionary
ERROR_MESSAGES = {
    SUCCESS: "Success",
    DIMENSION_MISMATCH: "Dimension mismatch",
    SINGULAR_MATRIX: "Singular matrix",
    ILL_CONDITIONED: "Ill-conditioned matrix",
    NUMERIC_ERROR: "Numeric error",
    INVALID_ARGUMENT: "Invalid argument",
    CONVERGENCE_ERROR: "Convergence failure",
    IMPLEMENTATION_ERROR: "Implementation error",
    UNKNOWN_ERROR: "Unknown error"
}

# Family and link function mappings
FAMILY = {
    'gaussian': 0,
    'bernoulli': 1,
    'poisson': 2
}

LINK = {
    'identity': 0,
    'logit': 1,
    'log': 2
}

# Load the library
_lib = _load_lib()

# ===============================================================
# Define function signatures for C functions
# ===============================================================

# Define fitGLM_c signature
_lib.fitGLM_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # X matrix (flattened)
    ctypes.c_int,                    # X rows
    ctypes.c_int,                    # X cols
    ctypes.POINTER(ctypes.c_double),  # y vector
    ctypes.c_int,                    # y size
    ctypes.c_int,                    # family code
    ctypes.c_int,                    # link code
    ctypes.c_double,                 # lambda (LASSO parameter)
    ctypes.c_int,                    # max iterations
    ctypes.c_double,                 # tolerance
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # output coefficients
    ctypes.POINTER(ctypes.c_int)     # output size
]
_lib.fitGLM_c.restype = ctypes.c_int

# Define memory management functions
_lib.freeGLMResult.argtypes = [ctypes.POINTER(ctypes.c_double)]
_lib.freeGLMResult.restype = None

# Define error handling function
_lib.getLastError_c.argtypes = [ctypes.POINTER(ctypes.c_int)]
_lib.getLastError_c.restype = ctypes.c_char_p

# Define matrix multiplication function
_lib.matrixMultiply_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # Matrix 1
    ctypes.c_int,                    # Rows 1
    ctypes.c_int,                    # Cols 1
    ctypes.POINTER(ctypes.c_double),  # Matrix 2
    ctypes.c_int,                    # Rows 2
    ctypes.c_int,                    # Cols 2
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # Result matrix
    ctypes.POINTER(ctypes.c_int),    # Result rows
    ctypes.POINTER(ctypes.c_int)     # Result cols
]
_lib.matrixMultiply_c.restype = ctypes.c_int

_lib.freeMatrix_c.argtypes = [ctypes.POINTER(ctypes.c_double)]
_lib.freeMatrix_c.restype = None

# ===============================================================
# Helper functions for error handling and data conversion
# ===============================================================

def _get_last_error():
    """Get the last error message from the library."""
    error_len = ctypes.c_int(0)
    error_msg_ptr = _lib.getLastError_c(ctypes.byref(error_len))
    if error_msg_ptr:
        error_msg = ctypes.string_at(error_msg_ptr).decode('utf-8')
        return error_msg
    return "Unknown error (no message available)"

def _handle_error(error_code):
    """Handle error codes and return appropriate error information."""
    if error_code == SUCCESS:
        return None
    
    error_msg = _get_last_error()
    default_msg = ERROR_MESSAGES.get(error_code, "Unknown error")
    
    return {
        'code': error_code,
        'message': error_msg if error_msg else default_msg
    }

def _check_family_link(family, link):
    """Convert string representations to integer codes if needed."""
    if isinstance(family, str):
        family = family.lower()
        if family not in FAMILY:
            raise ValueError(f"Unknown family: {family}. Use one of {list(FAMILY.keys())}")
        family = FAMILY[family]
    
    if isinstance(link, str):
        link = link.lower()
        if link not in LINK:
            raise ValueError(f"Unknown link: {link}. Use one of {list(LINK.keys())}")
        link = LINK[link]
    
    return family, link

# ===============================================================
# Main function wrappers
# ===============================================================

def fit_glm(X, y, family=0, link=0, lambda_val=0.0, max_iter=100, tol=1e-6):
    """
    Fit a generalized linear model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix with shape (n_samples, n_features)
    y : numpy.ndarray
        Response vector with shape (n_samples,)
    family : int or str
        Family code: 0=Gaussian, 1=Bernoulli, 2=Poisson
        Or string: 'gaussian', 'bernoulli', 'poisson'
    link : int or str
        Link function code: 0=identity, 1=logit, 2=log
        Or string: 'identity', 'logit', 'log'
    lambda_val : float
        LASSO regularization parameter (0 = no regularization)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'coefficients': numpy array of estimated coefficients (if successful)
        - 'error': error code (0 for success)
        - 'error_message': error message (if error occurred)
    """
    # Convert string family/link to integer codes if needed
    family, link = _check_family_link(family, link)
    
    # Ensure input arrays are contiguous and of the right type
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    # Check dimensions
    n_samples, n_features = X.shape
    if len(y) != n_samples:
        raise ValueError(f"X has {n_samples} samples, but y has {len(y)} elements")
    
    # Prepare C pointers for X and y
    x_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Prepare output pointers
    coefs_ptr = ctypes.POINTER(ctypes.c_double)()
    coefs_size = ctypes.c_int(0)
    
    # Call the C function
    error_code = _lib.fitGLM_c(
        x_ptr, 
        ctypes.c_int(n_samples), 
        ctypes.c_int(n_features), 
        y_ptr, 
        ctypes.c_int(len(y)), 
        ctypes.c_int(family), 
        ctypes.c_int(link), 
        ctypes.c_double(lambda_val), 
        ctypes.c_int(max_iter), 
        ctypes.c_double(tol),
        ctypes.byref(coefs_ptr),
        ctypes.byref(coefs_size)
    )
    
    # Check for errors
    error_info = _handle_error(error_code)
    if error_info:
        return {
            'error': error_code,
            'error_message': error_info['message'],
            'coefficients': None
        }
    
    # Copy coefficients to numpy array before freeing C memory
    coefficients = np.ctypeslib.as_array(coefs_ptr, shape=(coefs_size.value,)).copy()
    
    # Free the memory allocated by Haskell
    _lib.freeGLMResult(coefs_ptr)
    
    return {
        'error': SUCCESS,
        'error_message': None,
        'coefficients': coefficients
    }

def matrix_multiply(A, B):
    """
    Multiply two matrices using the Haskell library.
    
    Parameters:
    -----------
    A : numpy.ndarray
        First matrix
    B : numpy.ndarray
        Second matrix
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': numpy array of multiplication result (if successful)
        - 'error': error code (0 for success)
        - 'error_message': error message (if error occurred)
    """
    # Ensure inputs are contiguous and of the right type
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    
    # Get dimensions
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    
    # Prepare C pointers
    a_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Prepare output pointers
    result_ptr = ctypes.POINTER(ctypes.c_double)()
    result_rows = ctypes.c_int(0)
    result_cols = ctypes.c_int(0)
    
    # Call the C function
    error_code = _lib.matrixMultiply_c(
        a_ptr,
        ctypes.c_int(a_rows),
        ctypes.c_int(a_cols),
        b_ptr,
        ctypes.c_int(b_rows),
        ctypes.c_int(b_cols),
        ctypes.byref(result_ptr),
        ctypes.byref(result_rows),
        ctypes.byref(result_cols)
    )
    
    # Check for errors
    error_info = _handle_error(error_code)
    if error_info:
        return {
            'error': error_code,
            'error_message': error_info['message'],
            'result': None
        }
    
    # Copy result to numpy array before freeing C memory
    result = np.ctypeslib.as_array(
        result_ptr, 
        shape=(result_rows.value, result_cols.value)
    ).copy()
    
    # Free the memory allocated by Haskell
    _lib.freeMatrix_c(result_ptr)
    
    return {
        'error': SUCCESS,
        'error_message': None,
        'result': result
    }

def predict(X, coefficients, link=0):
    """
    Make predictions using a fitted GLM model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix with shape (n_samples, n_features)
    coefficients : numpy.ndarray
        Model coefficients
    link : int or str
        Link function code: 0=identity, 1=logit, 2=log
        Or string: 'identity', 'logit', 'log'
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    """
    # Convert string link to integer code if needed
    if isinstance(link, str):
        link = link.lower()
        if link not in LINK:
            raise ValueError(f"Unknown link: {link}. Use one of {list(LINK.keys())}")
        link = LINK[link]
    
    # Ensure input arrays are contiguous and of the right type
    X = np.ascontiguousarray(X, dtype=np.float64)
    
    # Calculate linear predictor (X @ coefficients)
    linear_predictor = X @ coefficients
    
    # Apply inverse link function
    if link == LINK['identity']:
        return linear_predictor
    elif link == LINK['logit']:
        # Inverse logit is sigmoid function: exp(x) / (1 + exp(x))
        exp_vals = np.exp(np.clip(linear_predictor, -30, 30))  # Clip to avoid overflow
        return exp_vals / (1.0 + exp_vals)
    elif link == LINK['log']:
        # Inverse log is exp
        return np.exp(linear_predictor)
    else:
        raise ValueError(f"Unsupported link function code: {link}")

# ===============================================================
# Scikit-learn compatible class interface
# ===============================================================

class GLM:
    """
    Generalized Linear Model with optional LASSO regularization.
    
    This class provides a scikit-learn compatible interface to the
    Haskell GLM implementation.
    
    Parameters:
    -----------
    family : str or int
        Distribution family ('gaussian', 'bernoulli', 'poisson' or integer code)
    link : str or int
        Link function ('identity', 'logit', 'log' or integer code)
    lambda_val : float, default=0.0
        LASSO regularization parameter (L1 penalty)
    max_iter : int, default=100
        Maximum number of iterations for optimization
    tol : float, default=1e-6
        Convergence tolerance
        
    Attributes:
    -----------
    coef_ : numpy.ndarray
        Model coefficients (available after fitting)
    n_iter_ : int
        Number of iterations (placeholder, currently not returned from C API)
    """
    
    def __init__(self, family='gaussian', link='identity', lambda_val=0.0, 
                 max_iter=100, tol=1e-6):
        self.family = family
        self.link = link
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.n_iter_ = None
        self._error = None
    
    def fit(self, X, y):
        """
        Fit the GLM model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Call the fit_glm function
        result = fit_glm(
            X, y, 
            family=self.family, 
            link=self.link, 
            lambda_val=self.lambda_val,
            max_iter=self.max_iter, 
            tol=self.tol
        )
        
        if result['error'] != SUCCESS:
            self._error = result['error_message']
            raise RuntimeError(f"GLM fitting failed: {result['error_message']}")
        
        self.coef_ = result['coefficients']
        self.n_iter_ = self.max_iter  # This could be updated if iterations were returned by C API
        self._error = None
        return self
    
    def predict(self, X):
        """
        Predict using the fitted GLM model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        C : array of shape (n_samples,)
            Predicted values
        """
        if self.coef_ is None:
            raise RuntimeError("This GLM instance is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        return predict(X, self.coef_, link=self.link)
    
    def score(self, X, y):
        """
        Return a score for the goodness of fit.
        
        For Gaussian models, this is R^2. For classification models, it's accuracy.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X
            
        Returns:
        --------
        score : float
            Mean accuracy for classification, R^2 for regression
        """
        if self.coef_ is None:
            raise RuntimeError("This GLM instance is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        
        # For Bernoulli, use classification accuracy
        if self.family == 'bernoulli' or self.family == 1:
            y_pred_binary = (y_pred >= 0.5).astype(int)
            return np.mean(y_pred_binary == y)
        
        # For Gaussian, use R^2 (coefficient of determination)
        if self.family == 'gaussian' or self.family == 0:
            u = ((y - y_pred) ** 2).sum()
            v = ((y - y.mean()) ** 2).sum()
            return 1 - (u / v)
        
        # For Poisson, use explained deviance
        # (This is a simplification - true deviance would be based on log-likelihood)
        return np.mean((y - y_pred) ** 2)
