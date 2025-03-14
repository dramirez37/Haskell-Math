"""
Synthetic Data Generation Module for GLM Unit Testing

This module provides functions to generate synthetic datasets for testing
Generalized Linear Model (GLM) implementations. It supports generating data
for Gaussian, Bernoulli, and Poisson distributions with appropriate link functions.
"""

import numpy as np


def generate_gaussian_data(n_samples, n_features, noise=0.1, seed=None):
    """
    Generate synthetic data for a linear model with Gaussian noise.
    
    This function creates a design matrix X and response vector y where:
    y = X @ beta + epsilon, with epsilon ~ N(0, noise)
    
    Parameters
    ----------
    n_samples : int
        Number of samples (rows in the design matrix)
    n_features : int
        Number of features (columns in the design matrix, excluding intercept)
    noise : float, default=0.1
        Standard deviation of Gaussian noise added to the response
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features+1)
        Design matrix with intercept column (first column is all 1's)
    y : ndarray of shape (n_samples,)
        Response vector
    beta : ndarray of shape (n_features+1,)
        True coefficients used to generate the data
        
    Examples
    --------
    >>> X, y, beta = generate_gaussian_data(100, 3, noise=0.2, seed=42)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}, beta: {beta}")
    X shape: (100, 4), y shape: (100,), beta: [...]
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate design matrix (add intercept column)
    X = np.column_stack((np.ones(n_samples), np.random.randn(n_samples, n_features)))
    
    # Generate true coefficients (including intercept)
    beta = np.random.uniform(-3, 3, size=n_features+1)
    
    # Calculate linear predictor (X @ beta)
    linear_pred = X @ beta
    
    # Add Gaussian noise
    y = linear_pred + np.random.normal(0, noise, size=n_samples)
    
    return X, y, beta


def generate_bernoulli_data(n_samples, n_features, probability=0.5, seed=None):
    """
    Generate synthetic data for a Bernoulli GLM (logistic regression).
    
    This function creates a design matrix X and binary response vector y where:
    p(y=1) = 1/(1+exp(-X @ beta))
    
    Parameters
    ----------
    n_samples : int
        Number of samples (rows in the design matrix)
    n_features : int
        Number of features (columns in the design matrix, excluding intercept)
    probability : float, default=0.5
        Controls the baseline probability; higher values increase the 
        proportion of positive cases
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features+1)
        Design matrix with intercept column (first column is all 1's)
    y : ndarray of shape (n_samples,)
        Binary response vector (0 or 1)
    beta : ndarray of shape (n_features+1,)
        True coefficients used to generate the data
        
    Examples
    --------
    >>> X, y, beta = generate_bernoulli_data(100, 3, probability=0.3, seed=42)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}, positive cases: {np.sum(y)}")
    X shape: (100, 4), y shape: (100,), positive cases: 32
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate design matrix (add intercept column)
    X = np.column_stack((np.ones(n_samples), np.random.randn(n_samples, n_features)))
    
    # Generate true coefficients (including intercept)
    beta = np.random.uniform(-2, 2, size=n_features+1)
    
    # Adjust intercept to control the baseline probability
    # This helps ensure we don't get extremely imbalanced datasets
    beta[0] = np.log(probability / (1 - probability))
    
    # Calculate linear predictor (X @ beta)
    linear_pred = X @ beta
    
    # Apply inverse logit link function to get probabilities
    probabilities = 1 / (1 + np.exp(-linear_pred))
    
    # Generate binary outcomes
    y = np.random.binomial(1, probabilities)
    
    return X, y, beta


def generate_poisson_data(n_samples, n_features, lambda_value=5.0, seed=None):
    """
    Generate synthetic data for a Poisson GLM (log-linear model).
    
    This function creates a design matrix X and count response vector y where:
    y ~ Poisson(exp(X @ beta))
    
    Parameters
    ----------
    n_samples : int
        Number of samples (rows in the design matrix)
    n_features : int
        Number of features (columns in the design matrix, excluding intercept)
    lambda_value : float, default=5.0
        Controls the baseline count rate; higher values increase the 
        expected counts
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features+1)
        Design matrix with intercept column (first column is all 1's)
    y : ndarray of shape (n_samples,)
        Count response vector (non-negative integers)
    beta : ndarray of shape (n_features+1,)
        True coefficients used to generate the data
        
    Examples
    --------
    >>> X, y, beta = generate_poisson_data(100, 3, lambda_value=3.0, seed=42)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}, mean count: {np.mean(y):.2f}")
    X shape: (100, 4), y shape: (100,), mean count: 2.87
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate design matrix (add intercept column)
    X = np.column_stack((np.ones(n_samples), np.random.randn(n_samples, n_features)))
    
    # Generate true coefficients (including intercept)
    beta = np.random.uniform(-1, 1, size=n_features+1)
    
    # Adjust intercept to control the baseline count rate
    beta[0] = np.log(lambda_value)
    
    # Calculate linear predictor (X @ beta)
    linear_pred = X @ beta
    
    # Apply exp link function to get expected counts (lambda)
    expected_counts = np.exp(linear_pred)
    
    # Generate count outcomes
    y = np.random.poisson(expected_counts)
    
    return X, y, beta


def validate_model_fit(X, y, true_beta, estimated_beta, family, tolerance=0.5):
    """
    Validate the accuracy of a GLM model by comparing estimated coefficients
    with true coefficients used to generate the data.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix
    y : ndarray of shape (n_samples,)
        Response vector
    true_beta : ndarray of shape (n_features,)
        True coefficients used to generate the data
    estimated_beta : ndarray of shape (n_features,)
        Estimated coefficients from the model
    family : str
        Distribution family ('gaussian', 'bernoulli', 'poisson')
    tolerance : float, default=0.5
        Maximum allowed average absolute difference between true and estimated coefficients
        
    Returns
    -------
    bool
        True if the model fit is accurate within the tolerance, False otherwise
    dict
        Dictionary with validation details
    """
    # Calculate absolute differences between true and estimated coefficients
    abs_diff = np.abs(true_beta - estimated_beta)
    avg_abs_diff = np.mean(abs_diff)
    
    # Calculate predictions
    if family == 'gaussian':
        y_pred = X @ estimated_beta
        r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        metrics = {
            'r_squared': r_squared,
            'mse': np.mean((y - y_pred)**2),
            'avg_coef_error': avg_abs_diff
        }
        
    elif family == 'bernoulli':
        linear_pred = X @ estimated_beta
        probabilities = 1 / (1 + np.exp(-linear_pred))
        y_pred = (probabilities >= 0.5).astype(int)
        accuracy = np.mean(y == y_pred)
        metrics = {
            'accuracy': accuracy,
            'avg_coef_error': avg_abs_diff
        }
        
    elif family == 'poisson':
        linear_pred = X @ estimated_beta
        expected_counts = np.exp(linear_pred)
        metrics = {
            'mean_abs_error': np.mean(np.abs(y - expected_counts)),
            'avg_coef_error': avg_abs_diff
        }
    
    else:
        raise ValueError(f"Unsupported family: {family}")
    
    # Validate if model fit is accurate within tolerance
    is_valid = avg_abs_diff <= tolerance
    
    return is_valid, metrics


if __name__ == "__main__":
    # Demonstrate usage of the synthetic data generation functions
    
    # Generate and print statistics for Gaussian data
    X_gauss, y_gauss, beta_gauss = generate_gaussian_data(
        n_samples=100, n_features=5, noise=0.2, seed=42)
    print("\nGaussian Data:")
    print(f"X shape: {X_gauss.shape}, y shape: {y_gauss.shape}")
    print(f"True coefficients: {beta_gauss}")
    print(f"y mean: {np.mean(y_gauss):.2f}, y std: {np.std(y_gauss):.2f}")
    
    # Generate and print statistics for Bernoulli data
    X_bern, y_bern, beta_bern = generate_bernoulli_data(
        n_samples=100, n_features=5, probability=0.3, seed=42)
    print("\nBernoulli Data:")
    print(f"X shape: {X_bern.shape}, y shape: {y_bern.shape}")
    print(f"True coefficients: {beta_bern}")
    print(f"Positive cases: {np.sum(y_bern)} ({np.mean(y_bern)*100:.1f}%)")
    
    # Generate and print statistics for Poisson data
    X_pois, y_pois, beta_pois = generate_poisson_data(
        n_samples=100, n_features=5, lambda_value=3.0, seed=42)
    print("\nPoisson Data:")
    print(f"X shape: {X_pois.shape}, y shape: {y_pois.shape}")
    print(f"True coefficients: {beta_pois}")
    print(f"y mean: {np.mean(y_pois):.2f}, y max: {np.max(y_pois)}")
    
    print("\nData generation complete.")
