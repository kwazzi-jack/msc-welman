import numpy as np
import logging
from source.data import load_data
from contextlib import contextmanager
from time import time
from datetime import datetime

def mean_square_error(A, B):
    res = (A - B)
    return np.mean(res * res.conjugate()).real   


def root_mean_square_error(A, B):
    res = (A - B).flatten()
    return np.sqrt(np.square(res).mean())


def flux_ratio(A, B):
    return np.divide(A, B)

def flux_error(A, B):
    return B * np.abs(1 - flux_ratio(A, B))

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
        results[i] = phi[-1].real

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

def calculate_mse_on_fluxes(params, paths):
    logging.debug("Calculating MSE on fluxes over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    true_data = load_data(params["paths"]["fluxes"]["true"]["files"][100])
    model = true_data["model"]
    Ix = true_data["Ix"]
    Iy = true_data["Iy"]
    true_flux = model[Ix, Iy]

    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        fluxes = load_data(path)["flux"]
        results[i] = mean_square_error(true_flux, fluxes)
    
    return results

def calculate_total_ratio_on_fluxes(params, paths):
    logging.debug("Calculating flux-ratio over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    true_data = load_data(params["paths"]["fluxes"]["true"]["files"][100])
    model = true_data["model"]
    Ix = true_data["Ix"]
    Iy = true_data["Iy"]
    true_flux = model[Ix, Iy]

    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        fluxes = load_data(path)["flux"]
        results[i] = total_flux_recovery_ratio(fluxes, true_flux)
    
    return results

def calculate_psi_0_on_fluxes(params, paths):
    logging.debug("Calculating psi-0 over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    true_data = load_data(params["paths"]["fluxes"]["true"]["files"][100])
    model = true_data["model"]
    Ix = true_data["Ix"]
    Iy = true_data["Iy"]
    true_flux = model[Ix, Iy]

    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        fluxes = load_data(path)["flux"]
        results[i] = mean_percent_flux_error(fluxes, true_flux)
    
    return results

def calculate_psi_1_on_fluxes(params, paths):
    logging.debug("Calculating psi-1 over various paths")
    if isinstance(paths, dict):
        paths = paths.values()
    
    true_data = load_data(params["paths"]["fluxes"]["true"]["files"][100])
    model = true_data["model"]
    Ix = true_data["Ix"]
    Iy = true_data["Iy"]
    true_flux = model[Ix, Iy]

    results = np.zeros((len(paths)), dtype=np.float64)
    for i, path in enumerate(paths):
        fluxes = load_data(path)["flux"]
        results[i] = weighted_mean_percent_flux_error(fluxes, true_flux)
    
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


def create_benchmarker(params, algorithm):
    paths = params["paths"]
    main_dir = paths["main"]["dir"]

    path = main_dir / f"benchmark.csv"

    if paths["main"]["files"]:
        paths["main"]["files"]["benchmark"] = path
    else:
        paths["main"]["files"] = {
            "benchmark": path
        }
    
    if "kalcal" in algorithm:
        parameter = "sigma_f"
        true_gains = load_data(paths["gains"]["true"]["files"])["gains"]
        steps = 2 * true_gains.shape[0]
    elif "quartical" in algorithm:
        parameter = "t_int"
        steps = params["quartical"]["iters"]
    else:
        parameter = "UNKNOWN"

    time_format = "%H:%M:%S"
    date_format = "%d/%m/%Y"

    if not path.exists():
        with open(path, "w") as file:
            file.write(f"date, time, id, algorithm, mp, parameter, value, " \
                       + "total_time, iterations, iter_time, total_memory\n")

    @contextmanager
    def benchmark(id, percent, value):
        try:
            start_date = datetime.now()
            start_time = time()
            yield 
            end_time = time()
            day_time = start_date.strftime(time_format)
            day_date = start_date.strftime(date_format)
            total_time = end_time - start_time
            iter_time = total_time/steps

            with open(path, "a") as file:
                file.write(f"{day_date}, {day_time}, {id}, {algorithm}, {percent}, " \
                        + f"{parameter}, {value}, {total_time}, {steps}, " \
                        + f"{iter_time}\n")
        except Exception as error:
            raise error
        
    return benchmark
