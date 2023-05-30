import numpy as np
  
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