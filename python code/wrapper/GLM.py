"""
Scikit-learn compatible interface for Generalized Linear Models (GLMs).

This module provides a Python interface to the Haskell GLM library,
allowing users to fit, evaluate, and make predictions with GLMs using
a familiar scikit-learn-like API.
"""

import numpy as np
from .PythonWrapper import fit_glm, predict as _predict, FAMILY, LINK

class GLM:
    """
    Generalized Linear Model with optional LASSO regularization.
    
    This class provides a scikit-learn compatible interface to the
    Haskell GLM implementation, supporting various distributions and
    link functions.
    
    Parameters
    ----------
    family : str or int
        Distribution family. Options:
        - 'gaussian' or 0: Gaussian (Normal) distribution
        - 'bernoulli' or 1: Bernoulli distribution for binary outcomes
        - 'poisson' or 2: Poisson distribution for count data
    
    link : str or int
        Link function. Options:
        - 'identity' or 0: Identity link (g(μ) = μ)
        - 'logit' or 1: Logit link (g(μ) = log(μ/(1-μ)))
        - 'log' or 2: Log link (g(μ) = log(μ))
        
        Note: If not specified, the canonical link for the chosen family will be used.
    
    lambda_val : float, default=0.0
        LASSO regularization parameter (L1 penalty)
    
    max_iter : int, default=100
        Maximum number of iterations for optimization
    
    tol : float, default=1e-6
        Convergence tolerance for coefficient changes
    
    verbose : bool, default=False
        Whether to print progress information during fitting
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the model.
        Available only after fit() is called.
    
    n_iter_ : int
        Number of iterations run during fitting.
        Available only after fit() is called.
    
    converged_ : bool
        Whether the algorithm converged.
        Available only after fit() is called.
    
    log_likelihood_ : float
        Final log-likelihood of the model.
        Available only after fit() is called.
    
    Examples
    --------
    >>> import numpy as np
    >>> from glm_wrapper import GLM
    >>> X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = GLM(family='bernoulli', link='logit')
    >>> model.fit(X, y)
    >>> print(model.coef_)
    >>> print(model.predict(X))
    """
    
    def __init__(self, family='gaussian', link=None, lambda_val=0.0, 
                 max_iter=100, tol=1e-6, verbose=False):
        """
        Initialize a GLM model.
        """
        self.family = family
        self.link = link
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Set default canonical link function if not provided
        if self.link is None:
            if self._get_family_code() == FAMILY['gaussian']:
                self.link = 'identity'
            elif self._get_family_code() == FAMILY['bernoulli']:
                self.link = 'logit'
            elif self._get_family_code() == FAMILY['poisson']:
                self.link = 'log'
        
        # Initialize attributes that will be set after fitting
        self.coef_ = None
        self.n_iter_ = None
        self.converged_ = None
        self.log_likelihood_ = None
        self._fitted = False
        
        # Store error information
        self._error = None
    
    def _get_family_code(self):
        """Convert family string or int to family code."""
        if isinstance(self.family, str):
            family_lower = self.family.lower()
            if family_lower not in FAMILY:
                raise ValueError(f"Unknown family: {self.family}. Use one of {list(FAMILY.keys())}")
            return FAMILY[family_lower]
        return self.family
    
    def _get_link_code(self):
        """Convert link string or int to link code."""
        if isinstance(self.link, str):
            link_lower = self.link.lower()
            if link_lower not in LINK:
                raise ValueError(f"Unknown link: {self.link}. Use one of {list(LINK.keys())}")
            return LINK[link_lower]
        return self.link
    
    def fit(self, X, y):
        """
        Fit the GLM model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        -------
        self : object
            Returns the instance itself for method chaining.
        
        Raises
        ------
        ValueError
            If the input data has incorrect shape or values.
        RuntimeError
            If the model fitting process fails.
        """
        # Convert input to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Dimension checks
        if X.ndim != 2:
            raise ValueError(f"X should be a 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y have incompatible shapes: X has {len(X)} samples, "
                             f"but y has {len(y)} samples.")
        
        # Call the Haskell GLM implementation through the FFI wrapper
        result = fit_glm(
            X, y,
            family=self._get_family_code(),
            link=self._get_link_code(),
            lambda_val=self.lambda_val,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        # Check for errors
        if result['error'] != 0:
            self._error = result['error_message']
            raise RuntimeError(f"GLM fitting failed: {result['error_message']}")
        
        # Store results
        self.coef_ = result['coefficients']
        self.converged_ = True  # We assume convergence if no error, could be extended
        self.n_iter_ = self.max_iter  # Currently the FFI doesn't return iterations
        self._fitted = True
        
        # Log likelihood is not currently returned through the FFI
        # This could be added if needed
        self.log_likelihood_ = None
        
        if self.verbose:
            print(f"Model fitting completed successfully.")
            print(f"Estimated coefficients: {self.coef_}")
        
        return self
    
    def predict(self, X, output_type='response'):
        """
        Predict using the fitted GLM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        output_type : str, default='response'
            Type of prediction:
            - 'response': Applies the inverse link function to get predictions on the
                         original scale of the response.
            - 'link': Returns the linear predictor values (X @ coef_).
            - 'class': For binary models, returns binary class predictions 
                      (thresholding at 0.5 for Bernoulli)
                      
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If output_type is invalid.
        """
        if not self._fitted:
            raise RuntimeError("This GLM instance is not fitted yet. Call 'fit' first.")
        
        # Convert input to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Compute linear predictor: η = Xβ
        linear_predictor = X @ self.coef_
        
        if output_type == 'link':
            return linear_predictor
        
        # Apply inverse link function to get predicted means
        response = _predict(X, self.coef_, link=self._get_link_code())
        
        if output_type == 'response':
            return response
        elif output_type == 'class':
            # For binary models, return class prediction
            if self._get_family_code() == FAMILY['bernoulli']:
                return (response >= 0.5).astype(int)
            else:
                raise ValueError("Class predictions are only available for binary (Bernoulli) models")
        else:
            raise ValueError(f"Invalid output_type: {output_type}. "
                             f"Use 'response', 'link', or 'class'.")
    
    def score(self, X, y):
        """
        Return a score for the goodness of fit.
        
        For Gaussian models, this is R-squared.
        For Bernoulli models, this is classification accuracy.
        For Poisson models, this is pseudo R-squared.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True values for X.
            
        Returns
        -------
        score : float
            Score value.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("This GLM instance is not fitted yet. Call 'fit' first.")
        
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Get predictions
        y_pred = self.predict(X)
        
        # For Bernoulli, use classification accuracy
        if self._get_family_code() == FAMILY['bernoulli']:
            y_pred_binary = (y_pred >= 0.5).astype(int)
            return np.mean(y_pred_binary == y)
        
        # For Gaussian, use R-squared (coefficient of determination)
        elif self._get_family_code() == FAMILY['gaussian']:
            u = np.sum((y - y_pred) ** 2)
            v = np.sum((y - np.mean(y)) ** 2)
            return 1 - (u / v) if v > 0 else 0.0
        
        # For Poisson, use pseudo R-squared
        elif self._get_family_code() == FAMILY['poisson']:
            y_pred_clip = np.clip(y_pred, 1e-10, None)  # Avoid log(0)
            
            # Deviance of fitted model
            dev_model = 2 * np.sum(y * np.log(np.clip(y, 1e-10, None) / y_pred_clip) - 
                                  (y - y_pred))
            
            # Deviance of null model (intercept only)
            y_mean = np.mean(y)
            dev_null = 2 * np.sum(y * np.log(np.clip(y, 1e-10, None) / y_mean) - 
                                 (y - y_mean))
            
            # Calculate pseudo R-squared
            return 1 - (dev_model / dev_null) if dev_null > 0 else 0.0
        
        # Default fallback
        return np.mean((y - y_pred) ** 2)
    
    def summary(self):
        """
        Generate a textual summary of the fitted model.
        
        Returns
        -------
        summary : str
            Summary of the fitted model including coefficients and fit statistics.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("This GLM instance is not fitted yet. Call 'fit' first.")
        
        # Get family and link names
        family_map = {FAMILY[k]: k for k in FAMILY}
        link_map = {LINK[k]: k for k in LINK}
        
        family_name = family_map.get(self._get_family_code(), "Unknown")
        link_name = link_map.get(self._get_link_code(), "Unknown")
        
        # Generate header
        header = (f"Generalized Linear Model (GLM) Results\n"
                 f"=====================================\n"
                 f"Family: {family_name.capitalize()}\n"
                 f"Link function: {link_name.capitalize()}\n"
                 f"Regularization (λ): {self.lambda_val}\n"
                 f"Number of samples: {self._n_samples if hasattr(self, '_n_samples') else 'Unknown'}\n"
                 f"Converged: {self.converged_}\n")
        
        if hasattr(self, 'log_likelihood_') and self.log_likelihood_ is not None:
            header += f"Log-likelihood: {self.log_likelihood_:.6f}\n"
        
        # Generate coefficients table
        coef_section = "\nCoefficients:\n"
        coef_section += "-" * 40 + "\n"
        coef_section += "    Parameter      Estimate\n"
        coef_section += "-" * 40 + "\n"
        
        for i, coef in enumerate(self.coef_):
            param_name = f"β{i}" if i > 0 else "Intercept" 
            coef_section += f"{param_name:>12}    {coef:10.6f}\n"
        
        coef_section += "-" * 40 + "\n"
        
        # Full summary
        summary_text = header + coef_section
        
        # Add warnings or notes if needed
        if self.lambda_val > 0:
            summary_text += "\nNote: LASSO regularization is active, coefficients are shrunk.\n"
        
        return summary_text
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'family': self.family,
            'link': self.link,
            'lambda_val': self.lambda_val,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'verbose': self.verbose
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **parameters : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
