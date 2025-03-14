"""
Exception handling for the GLM library's Python interface.

This module defines custom exception classes that map to Haskell-side errors
and provides utilities for converting error codes to appropriate Python exceptions.
"""

# Import error codes and helper function from the wrapper
from .PythonWrapper import (
    SUCCESS, DIMENSION_MISMATCH, SINGULAR_MATRIX, ILL_CONDITIONED,
    NUMERIC_ERROR, INVALID_ARGUMENT, CONVERGENCE_ERROR, IMPLEMENTATION_ERROR,
    UNKNOWN_ERROR, _get_last_error
)


class GLMError(Exception):
    """Base class for all GLM-related errors."""
    
    def __init__(self, message, error_code=None):
        """
        Initialize a GLM error.
        
        Parameters
        ----------
        message : str
            Detailed error message
        error_code : int, optional
            Numeric error code from the Haskell library
        """
        self.error_code = error_code
        super().__init__(message)


class DimensionMismatchError(GLMError):
    """
    Raised when input dimensions are incompatible.
    
    This typically occurs when matrix dimensions don't match requirements
    for operations like multiplication or when vector lengths are inconsistent.
    """
    pass


class SingularMatrixError(GLMError):
    """
    Raised when a matrix is singular or nearly singular.
    
    This occurs when matrix operations like inversion fail because
    the matrix is not full rank or is numerically close to singular.
    """
    pass


class IllConditionedError(GLMError):
    """
    Raised when a matrix is ill-conditioned.
    
    An ill-conditioned matrix may lead to numerical instability in computations.
    The condition number is typically very high (>1e15).
    """
    pass


class NumericError(GLMError):
    """
    Raised for general numerical errors.
    
    This includes issues like overflow, underflow, division by zero,
    or other numerical problems that occur during computation.
    """
    pass


class InvalidArgumentError(GLMError):
    """
    Raised when invalid arguments are provided to a function.
    
    This includes invalid parameter values, unsupported distribution families,
    or link functions that are incompatible with the chosen family.
    """
    pass


class ConvergenceError(GLMError):
    """
    Raised when an iterative algorithm fails to converge.
    
    This typically occurs in IRLS or coordinate descent when the
    maximum number of iterations is reached without meeting convergence criteria.
    """
    pass


class ImplementationError(GLMError):
    """
    Raised for internal implementation errors.
    
    These are errors that should not occur in normal operation and
    likely indicate a bug in the library implementation.
    """
    pass


def _raise_if_error(error_code, custom_message=None):
    """
    Check the error code from the Haskell FFI and raise a corresponding Python exception.
    
    Parameters
    ----------
    error_code : int
        Error code returned by Haskell FFI function
    custom_message : str, optional
        Additional context message to prepend to the error
    
    Raises
    ------
    GLMError or a subclass
        An appropriate exception based on the error code
    
    Notes
    -----
    If error_code is SUCCESS (0), this function does nothing and returns normally.
    """
    if error_code == SUCCESS:
        return
        
    # Get the error message from Haskell
    error_msg = _get_last_error()
    
    # Add custom message if provided
    if custom_message:
        error_msg = f"{custom_message}: {error_msg}"
    
    # Map error codes to specific exceptions
    if error_code == DIMENSION_MISMATCH:
        raise DimensionMismatchError(error_msg, error_code)
    elif error_code == SINGULAR_MATRIX:
        raise SingularMatrixError(error_msg, error_code)
    elif error_code == ILL_CONDITIONED:
        raise IllConditionedError(error_msg, error_code)
    elif error_code == NUMERIC_ERROR:
        raise NumericError(error_msg, error_code)
    elif error_code == INVALID_ARGUMENT:
        raise InvalidArgumentError(error_msg, error_code)
    elif error_code == CONVERGENCE_ERROR:
        raise ConvergenceError(error_msg, error_code)
    elif error_code == IMPLEMENTATION_ERROR:
        raise ImplementationError(error_msg, error_code)
    else:
        raise GLMError(f"Unknown error (code {error_code}): {error_msg}", error_code)


# Function to convert dictionary-based error results to exceptions
def raise_from_result(result, operation_name="Operation"):
    """
    Check a result dictionary for errors and raise exceptions if found.
    
    Many functions in the GLM wrapper return dictionaries with an 'error' key.
    This function checks for that key and raises appropriate exceptions.
    
    Parameters
    ----------
    result : dict
        Result dictionary with at least an 'error' key
    operation_name : str, optional
        Name of the operation for context in error messages
    
    Returns
    -------
    dict
        The original result dictionary if no errors were found
    
    Raises
    ------
    GLMError or a subclass
        An appropriate exception if the result indicates an error
    """
    if 'error' in result and result['error'] != SUCCESS:
        error_code = result['error']
        error_msg = result.get('error_message', _get_last_error())
        _raise_if_error(error_code, f"Error in {operation_name}")
    
    return result
