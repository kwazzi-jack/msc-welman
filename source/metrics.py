import numpy as np
import logging
from source.data import load_data


def mean_square_error(A, B):
    res = (A - B)
    return np.mean(res * res.conjugate()).real   


def root_mean_square_error(A, B):
    res = (A - B).flatten()
    return np.sqrt(np.square(res).mean())


def flux_ratio(A, B):
    return np.divide(A, B)


def percent_flux_error(A, B):
    return np.abs(1 - flux_ratio(A, B))


def total_flux_recovery_ratio(A, B):
    return np.divide(A.sum(), B.sum())


def mean_percent_flux_error(A, B):
    return percent_flux_error(A, B).mean()


def weighted_mean_percent_flux_error(A, B):
    return np.divide(np.multiply(B, percent_flux_error(A, B)).sum(), B.sum())

def get_energy_values(paths):
    logging.debug("Fetching energy function valyes from paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        phi = load_data(path)["phi"]
        results[i] = phi[-1]

    return results

def calculate_mse_on_gains(params, paths):
    logging.debug("Calculating MSE on gains over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    true_gains = load_data(params["paths"]["gains"]["true"]["files"])["gains"]

    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        gains = load_data(path)["gains"]
        results[i] = mean_square_error(true_gains, gains)
    
    return results

def calculate_rms_from_residuals(params, paths):
    logging.debug("Calculating RMS over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
        
    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        residual = load_data(path)["residual"]
        results[i] = np.sqrt(np.square(residual).mean())
    
    return results

def calculate_rms_from_corrected_residuals(params, paths):
    logging.debug("Calculating RMS over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
        
    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        corrected = load_data(path)["corrected"]
        results[i] = np.sqrt(np.square(corrected).mean())
    
    return results