from numba import njit
from casacore.tables import table
from source.parameters import refreeze, Settings, YamlDict
from pathlib import Path
from IPython.display import clear_output, display
from source.data import load_data
from source.other import check_for_data, DataExistsError, progress_bar
from source.metrics import mean_square_error
import numpy as np
import os
import logging
import time
from numpy.linalg import inv, cholesky
import subprocess
import zarr

@njit(fastmath=True, nogil=True, inline="always")
def compute_e_vector(data, model, weight, gains, ant1, ant2):
    """Calculate the error vector `e` from the widely
    linear form of the augmented `y - J^Hg` for a given time 
    step.

    Parameters
    ---------- 
    data : Complex128[:], (N_bl,)
        Slice of Measured Visibilities.
    model : Complex128[:], (N_bl,)
        Slice of Model Visibilities.
    weight : Complex128[:], (N_bl,)
        Slice of Visibility Weights.
    gains : Complex128[:], (N_ant,)
        Gains term to evaluate the Jacobians with.
    ant1 : Complex128[:], (N_bl,)
        Antenna Index Array 1.
    ant2: Complex128[:], (N_bl,)
        Antenna Index Array 2.
        
    Returns
    -------
    e : Complex128[:], (N_ant,)
        Error Vector of widely linear `y - J^Hg`.
    """
    
    # Result array
    e = np.zeros((model.shape[0],), dtype=gains.dtype)
    
    # Populate Array
    for b in range(model.shape[0]):
        p, q = ant1[b], ant2[b]
        
        # Contribution from normal Jacobian
        e[b] = data[b] \
            - gains[p] * model[b] * gains[q].conjugate()
    
    # Return with weights included
    return weight * e


@njit(fastmath=True, nogil=True, inline="always")
def compute_r_vector(error, model, weight, gains, ant1, ant2):
    """Calculate the residual vector `r` from the widely
    linear form of the augmented `J^He` for a given time 
    step.

    Parameters
    ---------- 
    error : Complex128[:], (N_bl,)
        Error between measurement vector and predicted
        gains estimate.
    model : Complex128[:], (N_bl,)
        Slice of Model Visibilities.
    weight : Complex128[:], (N_bl,)
        Slice of Visibility Weights.
    gains : Complex128[:], (N_ant,)
        Gains term to evaluate the Jacobians with.
    ant1 : Complex128[:], (N_bl,)
        Antenna Index Array 1.
    ant2: Complex128[:], (N_bl,)
        Antenna Index Array 2.
        
    Returns
    -------
    r : Complex128[:], (N_ant,)
        Residual Vector of a widely linear model.
    """
    
    # Result array
    r = np.zeros((gains.shape[0],), dtype=gains.dtype)
    
    # Populate array
    for b in range(model.shape[0]):
        p, q = ant1[b], ant2[b]
        
        # Contribution from normal Jacobian
        r[p] += weight[b] * gains[q] \
            * model[b].conjugate() * error[b] 
        
        # Contribution from conjugate Jacobian
        r[q] += weight[b] * gains[p] \
            * model[b] * error[b].conjugate()
    
    # Return with factor of 1/2 included
    return 1/2 * r


@njit(fastmath=True, nogil=True, inline="always")
def compute_u_vector(model, weight, gains, ant1, ant2):
    """Calculate the diagonal vector `u` from the widely
    linear form of the augmented `J^HJ` for a given time 
    step.

    Parameters
    ---------- 
    model : Complex128[:], (N_bl,)
        Slice of Model Visibilities.
    weight : Complex128[:], (N_bl,)
        Slice of Visibility Weights.
    gains : Complex128[:], (N_ant,)
        Gains term to evaluate the Jacobians with.
    ant1 : Complex128[:], (N_bl,)
        Antenna Index Array 1.
    ant2: Complex128[:], (N_bl,)
        Antenna Index Array 2.
        
    Returns
    -------
    u : Complex128[:], (N_ant,)
        Diagonal Vector of widely linear `J^HJ`.
    """
    
    # Result array
    u = np.zeros((gains.shape[0],), dtype=gains.dtype)
    
    # Populate array
    for b in range(model.shape[0]):
        p, q = ant1[b], ant2[b]
        
        # Contribution from normal Jacobian
        u[p] += weight[b] \
            * np.power(np.abs(model[b] * gains[q]), 2)
        
        # Contribution from conjugate Jacobian
        u[q] += weight[b] \
            * np.power(np.abs(model[b] * gains[p]), 2)
    
    # Return with factor of 1/2 included
    return 1/4 * u


@njit(fastmath=True, nogil=True, inline="always")
def compute_sinv_vector(model, weight, gains, tinv, ant1, ant2):
    """Calculate the diagonal vector $s^{-1}$ of inverse
    measurement covariance `S^{-1}` from widely linear 
    form Jacobians and `T^{-1}`. Computes `1 - JT^{-1}J^H`
    in widely linear form. Note, `T^{-1} = P`, same as
    filter covariance

    Parameters
    ---------- 
    model : Complex128[:], (N_bl,)
        Slice of Model Visibilities.
    weight : Complex128[:], (N_bl,)
        Slice of Visibility Weights.
    gains : Complex128[:], (N_ant,)
        Gains term to evaluate the Jacobians with.
    tinv : Complex128[:], (N_ant,)
        Prediction Precision [Vector]. <== Not sure on this labelling.
    ant1 : Complex128[:], (N_bl,)
        Antenna Index Array 1.
    ant2: Complex128[:], (N_bl,)
        Antenna Index Array 2.
        
    Returns
    -------
    sinv : Complex128[:], (N_bl,)
        Diagonal Vector of Inverse Measurement 
        Covariance. 
    """
    
    # Holder array
    temp = np.zeros((model.shape[0],), dtype=model.dtype)
    
    # Populate array
    for b in range(model.shape[0]):
        p, q = ant1[b], ant2[b]
        
        # Contribution from normal Jacobian
        temp[b] += np.power(np.abs(gains[q] * model[b]), 2) \
                * tinv[p]
        
        # Contribution from conjugate Jacobian
        temp[b] += np.power(np.abs(gains[p] * model[b]), 2) \
                * tinv[q]
    
    # Return result with weight and 1/2 included
    return 1.0 - 0.25 * weight * temp


@njit(fastmath=True, nogil=True)
def diag_filter(data, model, weight, mp, pp, q, calcPhi=False):
    """Filter algorithm of kalcal-diag that uses diagonal
    approximation and various optimizations to perform
    recursive gains calibration for full-complex, time-only
    calibration. The energy function calculation is optional 
    since the estimates do not depend on it.

    Parameters
    ----------
    data : Complex128[:], (N_row,)
        Measured Visibilities. 
    model : Complex128[:], (N_row,)
        Model Visibilities.
    weight : Complex128[:], (N_row,)
        Visibility Weights.
    mp : Complex128[:], (N_ant,)
        Prior Gains Estimate
    pp : Complex128[:], (N_ant,)
        Prior Gains Covariance [Vector].
    q : Complex128[:], (N_ant,)
        Process Noise Covariance [Vector].
    calcPhi : Bool, optional
        Flag whether the algorithm should calculate
        the energy function phi. Default is to
        skip this step.

    Returns
    -------
    m : Complex128[:, :], (N_time, N_ant)
        Filter Gains Estimates.
    p : Complex128[:, :], (N_time, N_ant)
        Filter Gains Covariances [Vector].
    phi : Complex128[:], (N_time,)
        Energy function calculation per step if
        `calcPhi` is true. Otherwise zeros.
    """
    
    # Calculate axis lengths
    n_ant = mp.size
    n_bl = n_ant * (n_ant - 1)//2
    n_time = data.size//n_bl
    
    # Generate antenna arrays (upper triangular indices)
    ant1, ant2 = np.triu_indices(n_ant, k=1)
    
    # Result arrays
    m = np.zeros((n_time, n_ant), dtype=mp.dtype)
    p = np.zeros((n_time, n_ant), dtype=pp.dtype)
    phi = np.zeros((n_time,), dtype=pp.dtype)
    
    # Introduce prior into the loop
    # Python Witchery
    m[-1], p[-1] = mp, pp
    
    # Run filter recursively
    for k in range(n_time):
        
        # Prediction Step
        mp, pp = m[k - 1], p[k - 1] + q
        
        # Extract slices
        bl_slice = slice(k * n_bl, (k + 1) * n_bl)
        data_slice = data[bl_slice]
        model_slice = model[bl_slice]
        weight_slice = weight[bl_slice]
        
        # Compute residual, J^Hr and diag(J^HJ), respectively.
        e = compute_e_vector(data_slice, model_slice, 
                            weight_slice, m[k - 1], ant1, ant2)
        r = compute_r_vector(e, model_slice, weight_slice, 
                             m[k - 1], ant1, ant2)
        u = compute_u_vector(model_slice, weight_slice, 
                             m[k - 1], ant1, ant2)
        
        # Update Step
        p[k] = 1.0/(1.0/pp + u)
        m[k] = mp + r * p[k]
        
        # If true, calculate phi_k
        if calcPhi:
            sinv = compute_sinv_vector(model_slice, weight_slice, 
                                       m[k - 1], p[k], ant1, ant2)            
            calc = np.sum(np.power(np.abs(e), 2) * sinv) \
                    - np.sum(np.log(sinv))            
            phi[k] += phi[k - 1] + 0.5 * calc

    # Return filter results
    return m, p, phi


@njit(fastmath=True, nogil=True)
def diag_smoother(m, p, q): 
    """Smoother algorithm of kalcal-diag that uses diagonal 
    approximation and various optimizations to perform 
    recursive gains calibration for full-complex, time-only 
    calibration given the filter results.

    Parameters
    ----------
    m : Complex128[:, :], (N_time, N_ant)
        Filter Gains Estimates.
    p : Complex128[:, :], (N_time, N_ant)
        Filter Gains Covariances [Vector].
    q : Complex128[:], (N_ant,)
        Process Noise Covariance [Vector].

    Returns
    -------
    ms : Complex128[:, :], (N_time, N_ant)
        Smoother Gains Estimates.
    ps : Complex128[:, :], (N_time, N_ant)
        Smoother Gains Covariances [Vector].
    """
    
    # Calculate axis lengths
    n_time = m.shape[0]
    
    # Result arrays
    ms = np.zeros_like(m)
    ps = np.zeros_like(p)
    
    # Introduce prior into the loop
    ms[-1], ps[-1] = m[-1], p[-1]
    
    # Run smoother recursively, skip last time step
    for k in range(-2, -(n_time + 1), -1):
        
        # Prediction step
        mp, pp = m[k], p[k] + q
        
        # Smoothing step
        w = p[k] / pp
        ms[k] = m[k] + w * (ms[k + 1] - mp)
        ps[k] = p[k] + np.power(w, 2) * (ps[k + 1] - pp)
    
    # Return smoother results
    return ms, ps


def kalcal_diag(msname, **kwargs):
    """Calibration algorithm to solve for full-complex
    time-only calibration solutions using the diagonal
    approximation with other optimizations as 
    described by the Wide Extended Kalman Filter and
    Smoother algorithms described in ch. 3. This
    algorithm calculates both filter and smoother 
    estimates and covariances, and both are saved to 
    file. Also, there is the option to calculate the
    energy function if need be.

    Parameters
    ----------
    msname : String
        Path to CASA Measurement Set.
    kwargs : Dict
        Dictionary of keyword arguments.
    
    Keywords
    --------
    vis_column : String
        Measured Visibilities Column Name.
    model_column : String
        Model Visibilities Column Name.
    sigma_f : Float64
        Process Noise [Standard Deviation] Parameter.
    calcPhi : Bool
        Flag filter algorithm to calculate 
        energy function terms.
    out_filter : String
        Path to Filter Results Output File.
    out_smoother : String
        Path to Smoother Results Output File.
    """
    
    # Retrieve visibility data
    with table(msname, ack=False) as tb:
        data = tb.getcol(kwargs["vis_column"])[..., 0, 0]
        model = tb.getcol(kwargs["model_column"])[..., 0, 0]
        weight = tb.getcol("WEIGHT")[..., 0]
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        
    # Calculate axis lengths
    n_ant = np.max((ant1.max(), ant2.max())) + 1
    n_bl = n_ant * (n_ant - 1)//2
    n_time = model.shape[0]/n_bl
    
    # Create priors
    sigma_f = kwargs["sigma_f"]
    mp = np.ones((n_ant,), dtype=np.complex128)
    pp = np.ones((n_ant,), dtype=np.complex128)
    q = sigma_f**2 * np.ones((n_ant,), dtype=np.complex128)
    
    # Run filter
    m, p, phi = diag_filter(data, model, weight, 
                                mp, pp, q, kwargs["calcPhi"])
    
    # Save filter results (codex format)
    with open(kwargs["out_filter"], "wb") as file:
        np.savez(file, 
                 gains=m,
                 var=p,
                 phi=phi
        )

    # Run smoother
    ms, ps = diag_smoother(m, p, q)   
       
    # Save smoother results (codex format)
    with open(kwargs["out_smoother"], "wb") as file:
        np.savez(file, 
                 gains=ms,
                 var=ps
        )

def kalcal_diag_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating kalcal-diag options config")
    paths = params["paths"]
    name = "kalcal-diag-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["kalcal-diag-config"] = str(Path(config_dir) / name)
    logging.debug(f"kalcal-diag config at: `{paths['kalcal-diag-config']}`")

    settings = Settings(
        name=name,
        header="Calibration run: <tt>kalcal-diag</tt>",
        description="""
        These options govern the line-search that will be set up
        for the rest of the experiment. This includes which process
        noise parameters to use, paths to solutions, and where to
        find to solutions.""",
        directory=str(config_dir),
        immutable_path=True,
    )

    settings["status"] = (
        "Algorithm Status",
        "The status of the algorithm, i.e., whether to use it or not.",
        ["ENABLED", "DISABLED"]
    )

    settings["n-points"] = (
        "Number of Runs",
        "How many runs, spaced across the interval, should be done.",
        32,
    )

    settings["prec"] = (
        "Precision of Process Noise",
        "How many decimals to use within the process noise values.",
        16
    )

    settings["low-bound"] = (
        "Exponent Lower-bound",
        "The lower-bound to use as an exponent for the smallest order of magnitude.",
        -4,
    )

    settings["up-bound"] = (
        "Exponent Upper-bound",
        "The upper-bound to use as an exponent for the largest order of magnitude.",
        -2,
    )

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("kalcal-diag settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("kalcal-diag settings set to default and saved")

    logging.debug("kalcal-diag settings complete and returning")
    return settings


def run_kalcal_diag_calibration(kal_diag_options, params, 
                                check_mse=False, progress=False):
    if progress:
        pbar = progress_bar("Runs")

    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    percents = params["percents"]

    # Parameters for log-search
    logging.info("Fetching kalcal-diag calibration run options")
    status = kal_diag_options["status"]
    n_points = kal_diag_options["n-points"]
    prec = kal_diag_options["prec"]
    lb = kal_diag_options["low-bound"]
    ub = kal_diag_options["up-bound"]

    if status == "DISABLED":
        logging.info("kalcal-diag is disabled, do nothing")
        logging.info("Updating parameter information")
        with refreeze(params) as file:
            file["kalcal-diag"] = {
                "status" : status,
                "n-points" : n_points,
                "prec" : prec,
                "low-bound": lb,
                "up-bound": ub
            }
        return
    
    sigma_fs = np.round(np.logspace(lb, ub, n_points), prec)
    logging.info("Calculated line search process noise parameters")

    try:
        filter_paths = paths["gains"]["kalcal-diag"]["filter"]
        smoother_paths = paths["gains"]["kalcal-diag"]["smoother"]
        filter_keys = filter_paths.keys()
        smoother_keys = smoother_paths.keys()

        for percent in percents:
            if percent not in filter_keys:
                raise KeyError
            
            if percent not in smoother_keys:
                raise KeyError
            
        for key in filter_keys:
            if key not in percents:
                raise KeyError

        for key in smoother_keys:
            if key not in percents:
                raise KeyError
        
        logging.debug("Filter and smoother paths match")

        try:
            files = os.listdir(filter_paths["dir"])
            if len(files):
                check_for_data(*files)
            files = os.listdir(smoother_paths["dir"])
            if len(files):
                check_for_data(*files)

            os.remove(filter_paths["dir"])
            while not os.path.exists(filter_paths["dir"]):
                time.sleep(0.1)
            os.mkdir(filter_paths["dir"])

            os.remove(smoother_paths["dir"])
            while not os.path.exists(smoother_paths["dir"]):
                time.sleep(0.1)
            os.mkdir(smoother_paths["dir"])
            logging.debug("Deleted filter and smoother gains")
        except DataExistsError:
            logging.info("No deletion done, returning.")
            return 
        logging.debug("Gains folders exist, cleaned folders.")
    except:
        logging.info("Updating path data")
        data_dir = paths["data-dir"]
        with refreeze(paths) as file:
            if file["gains"].get("kalcal-diag", True):
                file["gains"]["kalcal-diag"] = {
                    "dir" : data_dir / "gains" / "kalcal-diag"
                }        
                os.makedirs(paths["gains"]["kalcal-diag"]["dir"], 
                            exist_ok=True)
            kalcal_dir = file["gains"]["kalcal-diag"]

            if kalcal_dir.get("filter", True):
                file["gains"]["kalcal-diag"]["filter"] = {
                    "dir" : data_dir / "gains" / "kalcal-diag" / "filter"
                }
                os.makedirs(paths["gains"]["kalcal-diag"]["filter"]["dir"], 
                            exist_ok=True)

            if kalcal_dir.get("smoother", True):
                file["gains"]["kalcal-diag"]["smoother"] = {
                    "dir" : data_dir / "gains" / "kalcal-diag" / "smoother"
                }
                os.makedirs(paths["gains"]["kalcal-diag"]["smoother"]["dir"], 
                            exist_ok=True)
        logging.debug("Gains folders missing, created.")

    filter_dir = paths["gains"]["kalcal-diag"]["filter"]
    smoother_dir = paths["gains"]["kalcal-diag"]["smoother"]
    total_runs = n_points * len(percents)
    true_gains = load_data(paths["gains"]["true"])["true_gains"]
        
    if progress:
        pbar.total = total_runs

    logging.warning("May take long to start. `numba` is compiling the functions.")
    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    logging.info(rf"Using interval [1e{lb}, 1e{ub}].")
    for percent in percents:
        filter_paths = []
        smoother_paths = []
        for sigma_f in sigma_fs:
            filter_path = filter_dir["dir"] / \
                f"kalcal-diag-gains-filter-{percent}mp-sigma_f-{sigma_f}.npz"
            smoother_path = smoother_dir["dir"] / \
                f"kalcal-diag-gains-smoother-{percent}mp-sigma_f-{sigma_f}.npz" 

            start = time.time()
            kalcal_diag(
                str(paths["ms-path"]),
                vis_column="DATA_100MP",
                model_column=f"MODEL_{percent}MP",
                sigma_f=float(sigma_f),
                calcPhi=True,
                out_filter=filter_path,
                out_smoother=smoother_path
            )
            end = time.time()

            filter_paths.append(filter_path)
            smoother_paths.append(smoother_path)
            log_msg = f"kalcal-diag on {percent}MP with "\
                    + f"`sigma_f={sigma_f:.3e}`, {(end - start):.3g}s taken"
                         
            if check_mse:
                filter_gains = load_data(filter_path)["gains"]
                filter_mse = mean_square_error(true_gains, filter_gains)

                smoother_gains = load_data(smoother_path)["gains"]
                smoother_mse = mean_square_error(true_gains, smoother_gains)
                log_msg += f", with filter-MSE={filter_mse:.3g}, " \
                         + f"smoother-MSE={smoother_mse:.3g}"
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

        logging.debug(f"Saving gains results to files for {percent}MP")    
        with refreeze(paths) as file:
            filter_dir[percent] = filter_paths
            smoother_dir[percent] = smoother_paths   

    logging.info("Updating parameter information")
    with refreeze(params) as file:
        file["kalcal-diag"] = {
            "status" : status,
            "n-points" : n_points,
            "prec" : prec,
            "low-bound": lb,
            "up-bound": ub,
            "sigma-fs" : sigma_fs
        }


@njit(fastmath=True, nogil=True)
def cholesky_inv(X):
    """Perform matrix inverse using 
    cholesky decomposition."""
    L = cholesky(X)
    Linv = inv(L)
    return Linv.conjugate().T @ Linv


def isstacked(x):
    """Check if input vector is stacked."""
    n = x.size // 2
    return np.isclose(x[0:n], x[n:2*n].conj()).all()

    
def isaugmented(X):
    """Check if input matrix is augmented."""
    nx, ny = X.shape
    nx //= 2
    ny //= 2
    
    X11 = X[0:nx, 0:ny]
    X12 = X[0:nx, ny:2*ny]
    X21 = X[nx:2*nx, 0:ny]
    X22 = X[nx:2*nx, ny:2*ny]
    
    return np.isclose(X11, X22.conj()).all() and np.isclose(X12, X21.conj()).all()


@njit(fastmath=True, inline="always")
def stack_vector(x):
    """Stack the input vector with its conjugate
    form.
    
    Parameters
    ---------- 
    x : Complex128[:], (N_ant,) | (N_bl,)
        Input vector to stack.
        
    Returns
    -------
    Complex128[:], (2*N_ant,) | (2*N_bl,)
        Stacked input vector.
    """
    return np.hstack((x, x.conjugate()))


@njit(fastmath=True, inline="always", parallel=False)
def populate_jacobian(J, model, weight, ant1, ant2, gains):
    """Populate the augmented Jacobian evaluated with data
    and gains terms for the given time step.
    
    Parameters
    ---------- 
    J : Complex128[:, :], (2*N_bl, 2*N_ant)
        [Augmented] Jacobian term to populate.
    model : Complex128[:], (N_bl,)
        Slice of Model Visibilities.
    weight : Complex128[:], (N_bl,)
        Slice of Visibility Weights.
    gains : Complex128[:], (2*N_ant,)
        [Augmented] Gains term to evaluate 
        the Jacobians with.
    ant1 : Complex128[:], (N_bl,)
        Antenna Index Array 1.
    ant2: Complex128[:], (N_bl,)
        Antenna Index Array 2.
        
    Returns
    -------
    J : Complex128[:, :], (2*N_bl, 2*N_ant)
        Newly populated [Augmented] Jacobian term.
    """
    
    # Calculate axis lengths
    n_bl, n_ant = model.size, gains.size//2
    
    # Populate Array
    for b in range(n_bl):
        p, q = ant1[b], ant2[b]
        w = np.sqrt(weight[b])

        # Top Left block (normal Jacobian)
        J[b, p] = 1/2 * w * model[b] * gains[n_ant + q]

        # Top Right block (conjugate Jacobian)
        J[b, n_ant + q] = 1/2 * w * model[b] * gains[p]

        # Bottom Left block (conjugate Jacobian)
        J[n_bl + b, q] = 1/2 * w * model[b].conjugate() * gains[p + n_ant]

        # Bottom Right block (normal Jacobian)
        J[n_bl + b, n_ant + p] = 1/2 * w * model[b].conjugate() * gains[q]
    
    # Return populated augmented Jacobian
    return J
        
@njit(fastmath=True, nogil=True)
def full_filter(data, model, weight, mp, Pp, Q, calcPhi=False):
    """Filter algorithm of kalcal-full that uses the pure
    implementation and various optimizations to perform
    recursive gains calibration for full-complex, time-only
    calibration. The energy function calculation is optional 
    since the estimates do not depend on it.

    Parameters
    ----------
    data : Complex128[:], (N_row,)
        Measured Visibilities. 
    model : Complex128[:], (N_row,)
        Model Visibilities.
    weight : Complex128[:], (N_row,)
        Visibility Weights.
    mp : Complex128[:], (2*N_ant,)
        [Augmented] Prior Gains Estimate
    Pp : Complex128[:], (2*N_ant, 2*N_ant)
        [Augmented] Prior Gains Covariance Matrix.
    Q : Complex128[:], (2*N_ant, 2*N_ant)
        [Augmented] Process Noise Covariance Matrix.
    calcPhi : Bool, optional
        Flag whether the algorithm should calculate
        the energy function phi. Default is to
        skip this step.

    Returns
    -------
    m : Complex128[:, :], (N_time, 2*N_ant)
        [Augmented] Filter Gains Estimates.
    P : Complex128[:, :, :], (N_time, 2*N_ant, 2*N_ant)
        [Augmented] Filter Gains Covariances Matrices.
    phi : Complex128[:], (N_time,)
        Energy function calculation per step if
        `calcPhi` is true. Otherwise zeros.
    """
    
    # Calculate axis lengths
    n_ant = mp.size//2
    n_bl = n_ant * (n_ant - 1)//2
    n_time = data.size//n_bl
    
    # Generate antenna arrays (upper triangular indices)
    ant1, ant2 = np.triu_indices(n_ant, k=1)
    
    # Result arrays
    m = np.zeros((n_time, 2 * n_ant), dtype=np.complex128)
    P = np.zeros((n_time, 2 * n_ant, 2 * n_ant), dtype=np.complex128)
    phi = np.zeros((n_time,), dtype=np.complex128)
    
    # Data arrays
    J = np.zeros((2 * n_bl, 2 * n_ant), dtype=np.complex128)
    I = np.eye(2 * n_bl).astype(np.complex128)
    
    # Introduce prior into the loop
    m[-1], P[-1] = mp, Pp
    
    # Jitter term for matrix inversion stability
    jitter = 1e-8 * np.eye(2 * n_ant, dtype=np.complex128)
    
    # Run filter recursively
    for k in range(n_time):
        
        # Prediction Step
        mp, Pp = m[k - 1], P[k - 1] + Q
        
        # Extract slices
        bl_slice = slice(k * n_bl, (k + 1) * n_bl)
        data_slice = data[bl_slice]
        model_slice = model[bl_slice]
        weight_slice = weight[bl_slice]
        
        # Compute Jacobian, measurement vector and error vector
        J = populate_jacobian(J, model_slice, weight_slice, ant1, ant2, m[k - 1])
        v = stack_vector(np.sqrt(weight_slice) * data_slice)
        e = v - J @ mp
        
        # Update Step
        JH = J.conjugate().T
        T = cholesky_inv(Pp) + JH @ J + jitter
        Sinv = I - J @ cholesky_inv(T) @ JH
        K = Pp @ JH @ Sinv        
        m[k] = mp + K @ e
        P[k] = Pp - K @ J @ Pp
        
        # If true, calculate phi_k
        if calcPhi:
            l = np.diag(cholesky(Sinv))             
            calc = 0.5 * e.conjugate().T @ Sinv @ e - np.sum(np.log(l))
            phi[k] += phi[k - 1] + calc.real
        
    # Return filter results
    return m, P, phi


@njit(fastmath=True, nogil=True)
def full_smoother(m, P, Q):
    """Smoother algorithm of kalcal-full that uses the pure
    implementation and various optimizations to perform 
    recursive gains calibration for full-complex, time-only 
    calibration given the filter results.

    Parameters
    ----------
    m : Complex128[:, :], (N_time, 2*N_ant)
        [Augmented] Filter Gains Estimates.
    P : Complex128[:, :, :], (N_time, 2*N_ant, 2*N_ant)
        [Augmented] Filter Gains Covariance Matrices.
    Q : Complex128[:], (2*N_ant, 2*N_ant)
        [Augmented] Process Noise Covariance Matrix.

    Returns
    -------
    ms : Complex128[:, :], (N_time, 2*N_ant)
        [Augmented] Smoother Gains Estimates.
    Ps : Complex128[:, :, :], (N_time, 2*N_ant, 2*N_ant)
        [Augmented] Smoother Gains Covariance Matrices.
    """

    # Calculate axis lengths
    n_time = m.shape[0]
    
    # Result arrays
    ms = np.zeros_like(m)
    Ps = np.zeros_like(P)
    
    # Introduce prior into the loop
    ms[-1], Ps[-1] = m[-1], P[-1]
    
    # Run smoother recursively, skip last time step
    for k in range(-2, -(n_time + 1), -1):
        # Prediction step
        mp, Pp = m[k], P[k] + Q
        
        # Smoothing step
        Pinv = inv(Pp)
        W = P[k] @ Pinv
        ms[k] = m[k] + W @ (ms[k + 1] - mp)
        Ps[k] = P[k] + W @ (Ps[k + 1] - Pp) @ W.conjugate().T
    
    # Return smoother results
    return ms, Ps


def kalcal_full(msname, **kwargs):
    """Calibration algorithm to solve for full-complex
    time-only calibration solutions using the pure
    implementation with other optimizations as 
    described by the Augmented Extended Kalman Filter 
    and Smoother algorithms described in ch. 3. This
    algorithm calculates both filter and smoother 
    estimates and covariances, and both are saved to 
    file. Also, there is the option to calculate the
    energy function if need be.

    Parameters
    ----------
    msname : String
        Path to CASA Measurement Set.
    kwargs : Dict
        Dictionary of keyword arguments.
    
    Keywords
    --------
    vis_column : String
        Measured Visibilities Column Name.
    model_column : String
        Model Visibilities Column Name.
    sigma_f : Float64
        Process Noise [Standard Deviation] Parameter.
    calcPhi : Bool
        Flag filter algorithm to calculate 
        energy function terms.
    out_filter : String
        Path to Filter Results Output File.
    out_smoother : String
        Path to Smoother Results Output File.
    """
    
    # Retrieve visibility data
    with table(msname, ack=False) as tb:
        data = tb.getcol(kwargs["vis_column"])[..., 0, 0].astype(np.complex128)
        model = tb.getcol(kwargs["model_column"])[..., 0, 0].astype(np.complex128)
        weight = tb.getcol("WEIGHT")[..., 0].astype(np.complex128)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")

    # Calculate axis lengths
    n_ant = np.max((ant1.max(), ant2.max())) + 1
    n_bl = n_ant * (n_ant - 1)//2
    n_time = model.shape[0]//n_bl
    
    # Create priors and data terms
    sigma_f = kwargs["sigma_f"]
    mp = np.ones((2*n_ant,), dtype=np.complex128)
    Pp = np.eye(2*n_ant, dtype=np.complex128)
    Q = sigma_f**2 * np.eye(2*n_ant, dtype=np.complex128)
    J = np.zeros((2*n_bl, 2*n_ant), dtype=np.complex128)

    # Run filter
    m, P, phi = full_filter(data, model, weight, mp, Pp, Q, kwargs["calcPhi"])
    
    # Save filter results (codex format)
    with open(kwargs["out_filter"], "wb") as file:
        np.savez(file, 
                 gains=m[:, 0:n_ant],
                 var=P[:, 0:n_ant, 0:n_ant],
                 var2=P[:, 0:n_ant, n_ant:2*n_ant],
                 phi=phi
        )
    
    # Run smoother
    ms, Ps = full_smoother(m, P, Q)

    # Save smoother results (codex format)
    with open(kwargs["out_smoother"], "wb") as file:
        np.savez(file, 
                 gains=ms[:, 0:n_ant],
                 var=Ps[:, 0:n_ant, 0:n_ant],
                 var2=Ps[:, 0:n_ant, n_ant:2*n_ant]
        )

def kalcal_full_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating kalcal-full options config")
    paths = params["paths"]
    name = "kalcal-full-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["kalcal-full-config"] = str(Path(config_dir) / name)
    logging.debug(f"kalcal-full config at: `{paths['kalcal-full-config']}`")

    settings = Settings(
        name=name,
        header="Calibration run: <tt>kalcal-full</tt>",
        description="""
        These options govern the line-search that will be set up
        for the rest of the experiment. This includes which process
        noise parameters to use, paths to solutions, and where to
        find to solutions.""",
        directory=str(config_dir),
        immutable_path=True,
    )

    settings["status"] = (
        "Algorithm Status",
        "The status of the algorithm, i.e., whether to use it or not.",
        ["ENABLED", "DISABLED"]
    )

    settings["n-points"] = (
        "Number of Runs",
        "How many runs, spaced across the interval, should be done.",
        32,
    )

    settings["prec"] = (
        "Precision of Process Noise",
        "How many decimals to use within the process noise values.",
        16
    )

    settings["low-bound"] = (
        "Exponent Lower-bound",
        "The lower-bound to use as an exponent for the smallest order of magnitude.",
        -4,
    )

    settings["up-bound"] = (
        "Exponent Upper-bound",
        "The upper-bound to use as an exponent for the largest order of magnitude.",
        -2,
    )

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("kalcal-full settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("kalcal-full settings set to default and saved")

    logging.debug("kalcal-full settings complete and returning")
    return settings

def run_kalcal_full_calibration(kal_full_options, params, 
                                check_mse=False, progress=False):
    if progress:
        pbar = progress_bar("Runs")

    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    percents = params["percents"]

    # Parameters for log-search
    logging.info("Fetching kalcal-full calibration run options")
    status = kal_full_options["status"]
    n_points = kal_full_options["n-points"]
    prec = kal_full_options["prec"]
    lb = kal_full_options["low-bound"]
    ub = kal_full_options["up-bound"]

    if status == "DISABLED":
        logging.info("kalcal-full is disabled, do nothing")
        logging.info("Updating parameter information")
        with refreeze(params) as file:
            file["kalcal-diag"] = {
                "status" : status,
                "n-points" : n_points,
                "prec" : prec,
                "low-bound": lb,
                "up-bound": ub
            }
        return
    
    sigma_fs = np.round(np.logspace(lb, ub, n_points), prec)
    logging.info("Calculated line search process noise parameters")

    try:
        filter_paths = paths["gains"]["kalcal-full"]["filter"]
        smoother_paths = paths["gains"]["kalcal-full"]["smoother"]
        filter_keys = filter_paths.keys()
        smoother_keys = smoother_paths.keys()

        for percent in percents:
            if percent not in filter_keys:
                raise KeyError
            
            if percent not in smoother_keys:
                raise KeyError
            
        for key in filter_keys:
            if key not in percents:
                raise KeyError

        for key in smoother_keys:
            if key not in percents:
                raise KeyError
        
        logging.debug("Filter and smoother paths match")

        try:
            files = os.listdir(filter_paths["dir"])
            if len(files):
                check_for_data(*files)
            files = os.listdir(smoother_paths["dir"])
            if len(files):
                check_for_data(*files)

            os.remove(filter_paths["dir"])
            while not os.path.exists(filter_paths["dir"]):
                time.sleep(0.1)
            os.mkdir(filter_paths["dir"])

            os.remove(smoother_paths["dir"])
            while not os.path.exists(smoother_paths["dir"]):
                time.sleep(0.1)
            os.mkdir(smoother_paths["dir"])
            logging.debug("Deleted filter and smoother gains")
        except DataExistsError:
            logging.info("No deletion done, returning.")
            return 
        logging.debug("Gains folders exist, cleaned folders.")
    except:
        logging.info("Updating path data")
        data_dir = paths["data-dir"]
        with refreeze(paths) as file:
            if file["gains"].get("kalcal-full", True):
                file["gains"]["kalcal-full"] = {
                    "dir" : data_dir / "gains" / "kalcal-full"
                }        
                os.makedirs(paths["gains"]["kalcal-full"]["dir"], 
                            exist_ok=True)
            kalcal_dir = file["gains"]["kalcal-full"]

            if kalcal_dir.get("filter", True):
                file["gains"]["kalcal-full"]["filter"] = {
                    "dir" : data_dir / "gains" / "kalcal-full" / "filter"
                }
                os.makedirs(paths["gains"]["kalcal-full"]["filter"]["dir"], 
                            exist_ok=True)

            if kalcal_dir.get("smoother", True):
                file["gains"]["kalcal-full"]["smoother"] = {
                    "dir" : data_dir / "gains" / "kalcal-full" / "smoother"
                }
                os.makedirs(paths["gains"]["kalcal-full"]["smoother"]["dir"], 
                            exist_ok=True)
        logging.debug("Gains folders missing, created.")

    filter_dir = paths["gains"]["kalcal-full"]["filter"]
    smoother_dir = paths["gains"]["kalcal-full"]["smoother"]
    total_runs = n_points * len(percents)
    true_gains = load_data(paths["gains"]["true"])["true_gains"]
        
    if progress:
        pbar.total = total_runs

    logging.warning("May take long to start. `numba` is compiling the functions.")
    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    logging.info(rf"Using interval [1e{lb}, 1e{ub}].")
    for percent in percents:
        filter_paths = []
        smoother_paths = []
        for sigma_f in sigma_fs:
            filter_path = filter_dir["dir"] / \
                f"kalcal-full-gains-filter-{percent}mp-sigma_f-{sigma_f}.npz"
            smoother_path = smoother_dir["dir"] / \
                f"kalcal-full-gains-smoother-{percent}mp-sigma_f-{sigma_f}.npz" 

            start = time.time()
            kalcal_full(
                str(paths["ms-path"]),
                vis_column="DATA_100MP",
                model_column=f"MODEL_{percent}MP",
                sigma_f=float(sigma_f),
                calcPhi=True,
                out_filter=filter_path,
                out_smoother=smoother_path
            )
            end = time.time()

            filter_paths.append(filter_path)
            smoother_paths.append(smoother_path)
            log_msg = f"kalcal-full on {percent}MP with "\
                    + f"`sigma_f={sigma_f:.3e}`, {(end - start):.3g}s taken"
                         
            if check_mse:
                filter_gains = load_data(filter_path)["gains"]
                filter_mse = mean_square_error(true_gains, filter_gains)

                smoother_gains = load_data(smoother_path)["gains"]
                smoother_mse = mean_square_error(true_gains, smoother_gains)
                log_msg += f", with filter-MSE={filter_mse:.3g}, " \
                         + f"smoother-MSE={smoother_mse:.3g}"
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

        logging.debug(f"Saving gains results to files for {percent}MP")    
        with refreeze(paths) as file:
            filter_dir[percent] = filter_paths
            smoother_dir[percent] = smoother_paths   

    logging.info("Updating parameter information")
    with refreeze(params) as file:
        file["kalcal-full"] = {
            "status" : status,
            "n-points" : n_points,
            "prec" : prec,
            "low-bound": lb,
            "up-bound": ub,
            "sigma-fs" : sigma_fs
        }

def quartical_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating QuartiCal options config")
    paths = params["paths"]
    name = "quartical-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["quartical-config"] = str(Path(config_dir) / name)
    logging.debug(f"QuartiCal config at: `{paths['quartical-config']}`")

    settings = Settings(
        name=name,
        header="Calibration run: <tt>QuartiCal</tt>",
        description="""
        These options govern the line-search that will be set up
        for the rest of the experiment. This includes which time-
        interval sizes to use, paths to solutions, and where to
        find to solutions.""",
        directory=str(config_dir),
        immutable_path=True,
    )

    settings["status"] = (
        "Algorithm Status",
        "The status of the algorithm, i.e., whether to use it or not.",
        ["ENABLED", "DISABLED"]
    )

    settings["t-ints"] = (
        "Time Intervals",
        "Time intervals to use in the line-search as a comma-separated list. Slices are allowed.",
        "1:241",
    )

    settings["iters"] = (
        "Iterations",
        "Determines how many epochs to run for each estimation.",
        100,
    )

    settings["conv-crit"] = (
        "Convergence Criteria",
        "The lower threshold in which the algorithm estimation will aim for.",
        1e-7,
    )

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("QuartiCal settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("QuartiCal settings set to default and saved")

    logging.debug("QuartiCal settings complete and returning")
    return settings

def __identify_time_intervals(t_ints):
    logging.debug("Invoking function")
    result = []
    segments = t_ints.split(',')

    logging.debug("Finding time-intervals in input")
    for segment in segments:
        values = segment.strip().split(':')

        if len(values) == 1:
            result.append(int(values[0]))

        elif len(values) == 2:
            start = int(values[0])
            end = int(values[1]) + 1
            result.extend(range(start, end))

        elif len(values) == 3:
            start = int(values[0])
            end = int(values[1]) + 1
            step = int(values[2])
            result.extend(range(start, end, step))

    return np.array(result)


def quartical_setup(quart_options, params):
    logging.debug("Invoking function")
    logging.info("Updating path data")
    paths = params["paths"]
    config_dir = paths["config-dir"]
    config_path = config_dir / "quartical.yml"
    
    with refreeze(paths) as file:
        paths["quartical"] = config_path
    
    try:
        check_for_data(config_path)
    except:
        logging.info("No deletion done, return.")
        return
    
    logging.info("Fetching QuartiCal option information")
    status = quart_options["status"]
    t_ints = quart_options["t-ints"]
    t_ints = __identify_time_intervals(t_ints)
    iters = quart_options["iters"]
    conv_crit = quart_options["conv-crit"]

    with refreeze(params) as file:
        file["quartical"] = {
            "status" : status,
            "n-points" : len(t_ints),
            "t-ints" : t_ints,
            "iters" : iters,
            "conv-crit" : conv_crit
        }

    try:
        n_time = params["n-time"]
    except:
        n_time, n_ant = load_data(paths["gains"]["true"])["true_gains"].shape
        with refreeze(params) as file:            
            file["n-time"] = n_time
            file["n-ant"] = n_ant

    logging.info("Creating QuartiCal config")    
    config = YamlDict(config_path, overwrite=True)

    logging.debug("Generate `input_ms` options")
    config["input_ms"] = {
            "path": str(paths["ms-path"]),
            "data_column": "DATA_COLUMN",
            "sigma_column": "SIGMA_N",
            "time_chunk": 0,
            "select_corr": [0],
    }

    logging.debug("Generate `input_model` options")
    config["input_model"] = {
                "recipe": "MODEL_COLUMN",
            "invert_uvw": False,
            "source_chunks": int(n_time),
            "apply_p_jones": False
    }

    logging.debug("Generate `solver` options")
    config["solver"] = {
        "terms": ["G"],
        "iter_recipe": [iters],
        "robust": False,
        "threads": params["n-cpu"],
        "convergence_criteria": conv_crit,
        "propagate_flags": False
    }

    logging.debug("Generate `output` options")
    config["output"] = {
        "gain_directory": "/path/to/dir", 
        "log_to_terminal" : False,
        "overwrite": True,
        "flags": False,
        "apply_p_jones_inv": False
    }

    logging.debug("Generate `dask` options")
    config["dask"] = {
        "threads": 1
    }

    logging.debug("Generate `G` options")
    config["G"] = {
        "type": "diag_complex",
        "solve_per": "antenna",
        "direction_dependent": False,
        "time_interval": 8, 
        "freq_interval": 1
    }

    config.freeze()

    logging.info("Updating parameter data")
    with refreeze(params) as file:
        file["quartical"]["config"] = config


def convert_quartical_to_codex(gains_path, runs_path, t_int, paths):
    logging.debug("Invoking function")    
    logging.debug("Loading true-gains")
    true_gains = load_data(paths["gains"]["true"])["true_gains"]
    n_time, n_ant = true_gains.shape

    logging.debug(f"Open QuartiCal solutions at `{runs_path}`")
    codex_gains = np.zeros_like(true_gains)

    with zarr.open(runs_path, mode="r") as sol:
        quart_gains = sol.G.G_0.gains[:]

    logging.debug("Convert gains to codex format")
    for t in range(n_time):
        ti = t // t_int
        for a in range(n_ant):
            codex_gains[t, a]\
                = quart_gains[ti, 0, a, 0, 0]

    with open(gains_path, "wb") as file:
        np.savez(file, gains=codex_gains)


def run_quartical_calibration(quart_options, params, 
                                check_mse=False, progress=False):
    if progress:
        pbar = progress_bar("Runs")

    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    percents = params["percents"]

    # Parameters for log-search
    logging.info("Running QuartiCal setup")
    quartical_setup(quart_options, params)

    quart_params = params["quartical"]
    status = quart_params["status"]
    t_ints = quart_params["t-ints"]
    n_points = len(t_ints)

    if status == "DISABLED":
        logging.info("QuartiCal is disabled, do nothing")
        logging.info("Updating parameter information")
        with refreeze(params) as file:
            file["quartical"]["n-points"] = n_points
        return
    
    logging.info("Retrieved line search t-ints parameters")

    try:
        gains_paths = paths["gains"]["quartical"]
        gains_keys = gains_paths.keys()

        for percent in percents:
            if percent not in gains_keys:
                raise KeyError
            
        for key in gains_keys:
            if key not in percents:
                raise KeyError
        
        logging.debug("QuartiCal paths match")

        try:
            files = os.listdir(gains_paths["dir"])
            if len(files):
                check_for_data(*files)
            os.remove(gains_paths["dir"])
            while not os.path.exists(gains_paths["dir"]):
                time.sleep(0.1)
            os.mkdir(gains_paths["dir"])

            logging.debug("Deleted QuartiCal gains")
        except DataExistsError:
            logging.info("No deletion done, returning.")
            return 
        
        logging.debug("Gains folders exist, cleaned folders.")
    except:
        logging.info("Updating path data")
        data_dir = paths["data-dir"]
        with refreeze(paths) as file:
            if file["gains"].get("quartical", True):
                file["gains"]["quartical"] = {
                    "dir" : data_dir / "gains" / "quartical"
                }        
                os.makedirs(paths["gains"]["quartical"]["dir"], 
                            exist_ok=True)

        logging.debug("Gains folders missing, created.")

    gains_dir = paths["gains"]["quartical"]
    runs_dir = paths["runs-dir"]
    total_runs = n_points * len(percents)
    true_gains = load_data(paths["gains"]["true"])["true_gains"]
        
    if progress:
        pbar.total = total_runs

    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    for percent in percents:
        gains_paths = []
        for t_int in t_ints:
            gains_path = gains_dir["dir"] / \
                f"quartical-gains-{percent}mp-t_int-{t_int}.npz"
            runs_path = runs_dir / \
                f"quartical-run-{percent}mp-t_int-{t_int}" 

            model_column = f"MODEL_{percent}MP"
            vis_column = f"DATA_{percent}MP"
            quartical_args = [
                "goquartical", paths["quartical"],
                f"input_ms.data_column='{vis_column}'",
                f"input_model.recipe='{model_column}'",
                f"output.gain_directory='{runs_path}'",
                f"output.log_directory='{runs_path}'",
                f"G.time_interval={t_int}"
            ]
            
            start = time.time()
            with open(os.devnull, 'w') as fp:
                subprocess.Popen(quartical_args, 
                              stdout=fp, stderr=fp).wait()
            end = time.time()

            convert_quartical_to_codex(gains_path, runs_path, t_int, paths)
            gains_paths.append(gains_path)

            log_msg = f"QuartiCal on {percent}MP with "\
                    + f"`t-int={t_int}`, {(end - start):.3g}s taken"
                         
            if check_mse:
                gains = load_data(gains_path)["gains"]
                mse = mean_square_error(true_gains, gains)
                log_msg += f", with MSE={mse:.3g}"
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

        logging.debug(f"Saving gains results to files for {percent}MP")    
        with refreeze(paths) as file:
            gains_dir[percent] = gains_paths