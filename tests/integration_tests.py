"""
Integration Tests for GLM Library

This module provides integration tests for the GLM library by comparing its output
against established libraries like statsmodels and scikit-learn. It uses synthetic
datasets to fit models using both implementations and compares key outputs such as
coefficients, predictions, and performance metrics.
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

# Import our modules
from wrapper.GLM import GLM
from tests.synthetic_data import (
    generate_gaussian_data,
    generate_bernoulli_data,
    generate_poisson_data
)

# Helper functions for comparing models and reporting results
def compare_coefficients(glm_coef, ref_coef, tolerance=0.1):
    """
    Compare coefficients between our GLM and reference model.
    
    Parameters
    ----------
    glm_coef : ndarray
        Coefficients from our GLM model
    ref_coef : ndarray
        Coefficients from reference model
    tolerance : float, default=0.1
        Maximum allowed average absolute difference
        
    Returns
    -------
    bool
        True if the coefficients are similar within tolerance
    dict
        Dictionary with comparison details
    """
    abs_diff = np.abs(glm_coef - ref_coef)
    avg_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    
    is_similar = avg_diff <= tolerance
    
    return is_similar, {
        'average_absolute_difference': avg_diff,
        'maximum_absolute_difference': max_diff,
        'within_tolerance': is_similar
    }

def compare_predictions(glm_pred, ref_pred, tolerance=0.1):
    """
    Compare predictions between our GLM and reference model.
    
    Parameters
    ----------
    glm_pred : ndarray
        Predictions from our GLM model
    ref_pred : ndarray
        Predictions from reference model
    tolerance : float, default=0.1
        Maximum allowed average absolute difference
        
    Returns
    -------
    bool
        True if the predictions are similar within tolerance
    dict
        Dictionary with comparison details
    """
    abs_diff = np.abs(glm_pred - ref_pred)
    avg_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    
    is_similar = avg_diff <= tolerance
    
    return is_similar, {
        'average_absolute_difference': avg_diff,
        'maximum_absolute_difference': max_diff,
        'within_tolerance': is_similar
    }

def print_comparison_report(model_type, coef_results, pred_results, 
                           glm_metrics, ref_metrics, timings):
    """Print a formatted comparison report."""
    print(f"\n{'='*80}")
    print(f"INTEGRATION TEST RESULTS: {model_type.upper()} MODEL")
    print(f"{'='*80}")
    
    print("\nCOEFFICIENT COMPARISON:")
    print(f"  Average absolute difference: {coef_results['average_absolute_difference']:.6f}")
    print(f"  Maximum absolute difference: {coef_results['maximum_absolute_difference']:.6f}")
    print(f"  Within tolerance: {'YES' if coef_results['within_tolerance'] else 'NO'}")
    
    print("\nPREDICTION COMPARISON:")
    print(f"  Average absolute difference: {pred_results['average_absolute_difference']:.6f}")
    print(f"  Maximum absolute difference: {pred_results['maximum_absolute_difference']:.6f}")
    print(f"  Within tolerance: {'YES' if pred_results['within_tolerance'] else 'NO'}")
    
    print("\nPERFORMANCE METRICS:")
    for metric in glm_metrics:
        print(f"  {metric}:")
        print(f"    Our GLM:         {glm_metrics[metric]:.6f}")
        print(f"    Reference model: {ref_metrics[metric]:.6f}")
    
    print("\nEXECUTION TIME:")
    print(f"  Our GLM:         {timings['glm']:.6f} seconds")
    print(f"  Reference model: {timings['ref']:.6f} seconds")
    
    overall_result = coef_results['within_tolerance'] and pred_results['within_tolerance']
    print(f"\nTEST RESULT: {'PASSED' if overall_result else 'FAILED'}")
    print(f"{'='*80}\n")
    
    return overall_result

def test_gaussian_model(n_samples=1000, n_features=5, test_size=0.3, 
                        coef_tolerance=0.1, pred_tolerance=0.1, seed=42):
    """
    Test our GLM implementation against reference models for Gaussian data.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=5
        Number of features to generate
    test_size : float, default=0.3
        Proportion of data to use for testing
    coef_tolerance : float, default=0.1
        Tolerance for coefficient comparison
    pred_tolerance : float, default=0.1
        Tolerance for prediction comparison
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    bool
        True if the test passes, False otherwise
    """
    print("\nTesting Gaussian model (linear regression)...")
    
    # Generate synthetic data
    X, y, true_beta = generate_gaussian_data(n_samples, n_features, noise=0.2, seed=seed)
    
    # Split into train and test sets
    n_test = int(n_samples * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # Fit our GLM model
    glm_start = time.time()
    glm_model = GLM(family='gaussian', link='identity')
    glm_model.fit(X_train, y_train)
    glm_coef = glm_model.coef_
    glm_pred = glm_model.predict(X_test)
    glm_time = time.time() - glm_start
    
    # Calculate performance metrics for our GLM
    glm_mse = np.mean((y_test - glm_pred) ** 2)
    glm_r2 = r2_score(y_test, glm_pred)
    
    # Fit reference model (scikit-learn LinearRegression)
    ref_start = time.time()
    ref_model = LinearRegression(fit_intercept=False)  # No intercept since we already have a column of ones
    ref_model.fit(X_train, y_train)
    ref_coef = ref_model.coef_
    ref_pred = ref_model.predict(X_test)
    ref_time = time.time() - ref_start
    
    # Calculate performance metrics for reference model
    ref_mse = np.mean((y_test - ref_pred) ** 2)
    ref_r2 = r2_score(y_test, ref_pred)
    
    # Compare coefficients
    coef_similar, coef_results = compare_coefficients(glm_coef, ref_coef, tolerance=coef_tolerance)
    
    # Compare predictions
    pred_similar, pred_results = compare_predictions(glm_pred, ref_pred, tolerance=pred_tolerance)
    
    # Prepare metrics
    glm_metrics = {'MSE': glm_mse, 'R²': glm_r2}
    ref_metrics = {'MSE': ref_mse, 'R²': ref_r2}
    timings = {'glm': glm_time, 'ref': ref_time}
    
    # Print comparison report
    return print_comparison_report('Gaussian', coef_results, pred_results, 
                                  glm_metrics, ref_metrics, timings)

def test_bernoulli_model(n_samples=1000, n_features=5, test_size=0.3, 
                         coef_tolerance=0.2, pred_tolerance=0.1, seed=42):
    """
    Test our GLM implementation against reference models for Bernoulli data.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=5
        Number of features to generate
    test_size : float, default=0.3
        Proportion of data to use for testing
    coef_tolerance : float, default=0.2
        Tolerance for coefficient comparison
    pred_tolerance : float, default=0.1
        Tolerance for prediction comparison
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    bool
        True if the test passes, False otherwise
    """
    print("\nTesting Bernoulli model (logistic regression)...")
    
    # Generate synthetic data
    X, y, true_beta = generate_bernoulli_data(n_samples, n_features, probability=0.5, seed=seed)
    
    # Split into train and test sets
    n_test = int(n_samples * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # Fit our GLM model
    glm_start = time.time()
    glm_model = GLM(family='bernoulli', link='logit')
    glm_model.fit(X_train, y_train)
    glm_coef = glm_model.coef_
    glm_pred_prob = glm_model.predict(X_test)
    glm_pred_class = (glm_pred_prob > 0.5).astype(int)
    glm_time = time.time() - glm_start
    
    # Calculate performance metrics for our GLM
    glm_accuracy = accuracy_score(y_test, glm_pred_class)
    glm_log_loss = -np.mean(y_test * np.log(glm_pred_prob + 1e-10) + 
                           (1 - y_test) * np.log(1 - glm_pred_prob + 1e-10))
    
    # Fit reference model (scikit-learn LogisticRegression)
    ref_start = time.time()
    ref_model = LogisticRegression(fit_intercept=False, penalty='none', solver='lbfgs', max_iter=1000)
    ref_model.fit(X_train, y_train)
    ref_coef = ref_model.coef_[0]  # LogisticRegression returns a 2D array
    ref_pred_prob = ref_model.predict_proba(X_test)[:, 1]
    ref_pred_class = (ref_pred_prob > 0.5).astype(int)
    ref_time = time.time() - ref_start
    
    # Calculate performance metrics for reference model
    ref_accuracy = accuracy_score(y_test, ref_pred_class)
    ref_log_loss = -np.mean(y_test * np.log(ref_pred_prob + 1e-10) + 
                           (1 - y_test) * np.log(1 - ref_pred_prob + 1e-10))
    
    # Compare coefficients
    coef_similar, coef_results = compare_coefficients(glm_coef, ref_coef, tolerance=coef_tolerance)
    
    # Compare probability predictions
    pred_similar, pred_results = compare_predictions(glm_pred_prob, ref_pred_prob, tolerance=pred_tolerance)
    
    # Prepare metrics
    glm_metrics = {'Accuracy': glm_accuracy, 'Log Loss': glm_log_loss}
    ref_metrics = {'Accuracy': ref_accuracy, 'Log Loss': ref_log_loss}
    timings = {'glm': glm_time, 'ref': ref_time}
    
    # Print comparison report
    return print_comparison_report('Bernoulli', coef_results, pred_results, 
                                  glm_metrics, ref_metrics, timings)

def test_poisson_model(n_samples=1000, n_features=5, test_size=0.3, 
                       coef_tolerance=0.2, pred_tolerance=0.2, seed=42):
    """
    Test our GLM implementation against reference models for Poisson data.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=5
        Number of features to generate
    test_size : float, default=0.3
        Proportion of data to use for testing
    coef_tolerance : float, default=0.2
        Tolerance for coefficient comparison
    pred_tolerance : float, default=0.2
        Tolerance for prediction comparison
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    bool
        True if the test passes, False otherwise
    """
    print("\nTesting Poisson model (log-linear model)...")
    
    # Generate synthetic data
    X, y, true_beta = generate_poisson_data(n_samples, n_features, lambda_value=3.0, seed=seed)
    
    # Split into train and test sets
    n_test = int(n_samples * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # Fit our GLM model
    glm_start = time.time()
    glm_model = GLM(family='poisson', link='log')
    glm_model.fit(X_train, y_train)
    glm_coef = glm_model.coef_
    glm_pred = glm_model.predict(X_test)
    glm_time = time.time() - glm_start
    
    # Calculate performance metrics for our GLM
    glm_mae = mean_absolute_error(y_test, glm_pred)
    # Calculate pseudo R^2 for Poisson model
    null_model = np.mean(y_train) * np.ones_like(y_test)
    null_deviance = 2 * np.sum(y_test * np.log(y_test / (null_model + 1e-10) + 1e-10) - (y_test - null_model))
    model_deviance = 2 * np.sum(y_test * np.log(y_test / (glm_pred + 1e-10) + 1e-10) - (y_test - glm_pred))
    glm_pseudo_r2 = 1 - (model_deviance / null_deviance) if null_deviance != 0 else 0
    
    # Fit reference model (statsmodels GLM with Poisson family)
    ref_start = time.time()
    sm_X_train = sm.add_constant(X_train[:, 1:])  # Add constant for statsmodels but remove first column
    sm_X_test = sm.add_constant(X_test[:, 1:])    # since we already have one in our data
    
    ref_model = sm.GLM(y_train, sm_X_train, family=sm.families.Poisson())
    ref_results = ref_model.fit()
    ref_coef = ref_results.params
    ref_pred = ref_results.predict(sm_X_test)
    ref_time = time.time() - ref_start
    
    # Calculate performance metrics for reference model
    ref_mae = mean_absolute_error(y_test, ref_pred)
    # Calculate pseudo R^2 for reference model
    model_deviance_ref = 2 * np.sum(y_test * np.log(y_test / (ref_pred + 1e-10) + 1e-10) - (y_test - ref_pred))
    ref_pseudo_r2 = 1 - (model_deviance_ref / null_deviance) if null_deviance != 0 else 0
    
    # Compare coefficients
    coef_similar, coef_results = compare_coefficients(glm_coef, ref_coef, tolerance=coef_tolerance)
    
    # Compare predictions
    pred_similar, pred_results = compare_predictions(glm_pred, ref_pred, tolerance=pred_tolerance)
    
    # Prepare metrics
    glm_metrics = {'MAE': glm_mae, 'Pseudo R²': glm_pseudo_r2}
    ref_metrics = {'MAE': ref_mae, 'Pseudo R²': ref_pseudo_r2}
    timings = {'glm': glm_time, 'ref': ref_time}
    
    # Print comparison report
    return print_comparison_report('Poisson', coef_results, pred_results, 
                                  glm_metrics, ref_metrics, timings)

if __name__ == "__main__":
    print("=" * 80)
    print("GLM INTEGRATION TEST SUITE")
    print("Comparing our GLM implementation against established libraries")
    print("=" * 80)
    
    # Set parameters for all tests
    n_samples = 1000
    n_features = 5
    test_size = 0.3
    seed = 42
    
    # Start timing
    start_time = time.time()
    
    # Run tests
    gaussian_result = test_gaussian_model(n_samples=n_samples, n_features=n_features, 
                                         test_size=test_size, seed=seed)
    
    bernoulli_result = test_bernoulli_model(n_samples=n_samples, n_features=n_features, 
                                           test_size=test_size, seed=seed)
    
    poisson_result = test_poisson_model(n_samples=n_samples, n_features=n_features, 
                                       test_size=test_size, seed=seed)
    
    # Print overall results
    print("\n" + "=" * 80)
    print("OVERALL TEST RESULTS")
    print("=" * 80)
    
    all_passed = gaussian_result and bernoulli_result and poisson_result
    print(f"Gaussian Test:  {'PASSED' if gaussian_result else 'FAILED'}")
    print(f"Bernoulli Test: {'PASSED' if bernoulli_result else 'FAILED'}")
    print(f"Poisson Test:   {'PASSED' if poisson_result else 'FAILED'}")
    print("-" * 80)
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
