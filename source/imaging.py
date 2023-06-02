import numpy as np
from africanus.calibration.utils import corrupt_vis, correct_vis
from ducc0.wgridder import ms2dirty, dirty2ms
import logging
from casacore.tables import table, maketabdesc, makearrcoldesc
from pfb.utils.fits import set_wcs, save_fits
from source.data import load_data
from source.metrics import mean_square_error, root_mean_square_error
from source.parameters import refreeze
from source.other import progress_bar, check_for_data, DataExistsError
import os
import subprocess
import time

def calculate_image_noise_rms(params):
    logging.debug("Invoking function")
    paths = params["paths"]

    logging.debug("Fetching table information")
    with table(str(paths["ms-path"]), ack=False) as tb:
        noise = tb.getcol("NOISE_100MP")

    logging.debug("Calculating RMS noise from `NOISE_100MP`.")
    return np.sqrt((noise * noise.conj()).mean()).real

def calculate_imaging_columns(path, params):
    logging.debug("Invoking function")
    paths = params["paths"]

    logging.debug("Fetching table information")
    with table(str(paths["ms-path"]), ack=False) as tb:
        TIME = tb.getcol("TIME")
        ANT1 = tb.getcol("ANTENNA1")
        ANT2 = tb.getcol("ANTENNA2")
        vis = tb.getcol("DATA_100MP")
        flag = tb.getcol("FLAG")
        weights = tb.getcol("WEIGHT")

    _, tbin_indices, tbin_counts = np.unique(TIME,
                                             return_index=True,
                                             return_counts=True)
    
    logging.debug(f"Correcting visibility data with gains in `{path}`")
    gains = load_data(path)["true_gains"][..., None, None, None]
    corrected_data = correct_vis(
        tbin_indices,
        tbin_counts,
        ANT1,
        ANT2,
        gains,
        vis,
        flag
    )

    logging.debug("Calculating weight spectrum")
    abs_sqr_gains = np.power(np.abs(gains), 2)
    weight_spectrum = corrupt_vis(tbin_indices, tbin_counts, 
                                  ANT1, ANT2, abs_sqr_gains, 
                                  weights[..., None, None]).real
     
    with table(str(paths["ms-path"]), readonly=False, ack=False) as tb:
        desc = maketabdesc(makearrcoldesc(
                "WEIGHT_SPECTRUM",
                weight_spectrum[0, 0, 0],
                ndim=2,
                shape=[1, 1],
                valuetype="float"
        ))

        dminfo = tb.getdminfo("DATA_100MP")
        dminfo["NAME"] = f"weight_spectrum"
        # tb.removecols("WEIGHT_SPECTRUM")
        
        try:
            tb.addcols(desc, dminfo)
        except:
            pass
        
        logging.debug("Saving imaging data to table")
        tb.putcol("CORRECTED_DATA", corrected_data)
        tb.putcol("WEIGHT_SPECTRUM", weight_spectrum.astype(np.float32))

def image_to_fits(new_image, cell_rad, fits_file, freq, radec):
    nx, ny = new_image.shape
    header = set_wcs(cell_rad, cell_rad, nx, ny, radec, freq)
    save_fits(fits_file, new_image, header)
    
    
def pcg(A, b, x0, M=None, tol=1e-5, maxit=500, 
            report_freq=10, verbose=False):
    
    if M is None:
        def M(x): return x
    
    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = np.vdot(r, y)
    
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
        
    k = 0
    x = x0
    eps = 1.0
    stall_count = 0
    
    while eps > tol and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        beta = rnorm_next / rnorm
        p = beta * p - y
        rnorm = rnorm_next
        k += 1
        eps = rnorm / eps0
        
        if not k % report_freq and verbose:
           print("At iteration %i eps = %f" % (k, eps))

    if k >= maxit and verbose:
        print(f"Max iters reached at k = {k}. eps = {eps}.")
    elif verbose:
        print("Success, converged after %i iters" % k)
    return x


def calculate_recovered_flux(params, gains_file, vis_column, 
                                model_file, fluxes_file, 
                                residual_file, dirty_file):
    logging.debug("Invoking function")
    logging.debug("Fetching parameters")
    paths = params["paths"]
    ms_path = paths["data"]["ms"]
    with table(str(ms_path), ack=False) as tb:
        UVW = tb.getcol("UVW")
        TIME = tb.getcol("TIME")
        ANT1 = tb.getcol("ANTENNA1")
        ANT2 = tb.getcol("ANTENNA2")
        vis = tb.getcol(vis_column)[..., 0].astype(np.complex128)

    with table(f"{ms_path}::FIELD", ack=False) as tb:
        RADEC = tb.getcol('PHASE_DIR')[0][0]     

    with table(f"{ms_path}::SPECTRAL_WINDOW", ack=False) as tb:
        FREQ = tb.getcol("CHAN_FREQ").flatten()
    
    # Retrieve model image, pixel-size, and cell-size
    skymodel = load_data(model_file)
    model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    cell_rad = skymodel["cell_rad"]
    
    # Image shape
    nx, ny = model.shape
    
    # Create mask of source locations from model image
    mask = np.where(model, 1.0, 0)
    
    # Retrieve gains, but account for kalcal extra axis
    gains = np.load(gains_file)["gains"][..., None, None, None]

    # Time bin indices and counts
    _, tbin_indices, tbin_counts = np.unique(TIME, return_index=True,
                                            return_counts=True)
    
    # V = Jp int I kpq dl dm/n Jq.H
    # V = G R mask x   G = Mueller term,  G = Jp Jq.H,  G.H G = Jq Jq.H Jp.H Jp
    G = corrupt_vis(tbin_indices, tbin_counts, ANT1, ANT2,
                    gains, np.ones_like(vis[..., None, None]))[:, :, 0]
    
    # Sigma Inverse and Weights
    sigma_n = params["sigma-n"]
    tol = params["tol"]
    n_cpu = params["n-cpu"]
    S = 1/(2*sigma_n**2)
    W = (S * G.conj() * G).real
    logging.debug("Weighting calculated")

    # x = (R.H G.H G R)inv R.H G.H V
    dirty_image = ms2dirty(uvw=UVW, freq=FREQ, ms=S * G.conj() * vis,
                     npix_x=nx, npix_y=ny,
                     pixsize_x=cell_rad, pixsize_y=cell_rad,
                     epsilon=tol, nthreads=n_cpu, do_wstacking=True)
    logging.debug("Dirty image created")

    def hess(x):
        tmp = dirty2ms(uvw=UVW, freq=FREQ, dirty=mask * x,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=tol, nthreads=n_cpu, do_wstacking=True)
        
        res = ms2dirty(uvw=UVW, freq=FREQ, ms=tmp, wgt=W,
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=tol, nthreads=n_cpu, do_wstacking=True)
        
        return mask * res

    logging.debug("Running flux-extractor")
    recovered_image = pcg(hess, mask * dirty_image, x0=mask, tol=tol, verbose=False)

    # Retrieve flux per source based on positions
    recovered_flux = recovered_image[Ix, Iy]
    
    recovered_vis = dirty2ms(uvw=UVW, freq=FREQ, dirty=recovered_image,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    epsilon=tol, nthreads=n_cpu, do_wstacking=True)
    logging.debug("Corrected-residual image created")

    residual_image = ms2dirty(uvw=UVW, freq=FREQ, ms=S * G.conj() * (vis - G * recovered_vis),
                    npix_x=nx, npix_y=ny,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    epsilon=tol, nthreads=n_cpu, do_wstacking=True)
    
    logging.debug("Recovered flux found, saved to file")
    residual_image /= W.sum()
    # Save flux per source to file
    with open(fluxes_file, "wb") as file:
        np.savez(file, 
                 flux=recovered_flux,
                 residual=residual_image
                )
    image_to_fits(residual_image, cell_rad, residual_file, FREQ[0], RADEC)

    dirty_image /= W.sum()
    image_to_fits(dirty_image, cell_rad, dirty_file, FREQ[0], RADEC)
    logging.debug("Fits files saved")


def run_quartical_flux_extractor(params, progress=False, check_metric=False):
    if progress:
        pbar = progress_bar("Images")

    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    percents = params["percents"]

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
    logging.info("Updating path data")
    fluxes_paths = paths["fluxes"]["quartical"]["files"]
    fluxes_dir = paths["fluxes"]["quartical"]["dir"]
    fluxes_template = paths["fluxes"]["quartical"]["template"]

    fits_paths = paths["fits"]["quartical"]["files"]
    fits_dir = paths["fits"]["quartical"]["dir"]
    fits_template = paths["fits"]["quartical"]["template"]

    for percent in percents:
        fluxes_paths[percent] = {}
        fits_paths[percent] = {}
        for t_int in t_ints:
            fluxes_paths[percent][t_int] = fluxes_dir / fluxes_template.format(
                                           mp=percent, t_int=t_int)            
            fits_paths[percent][t_int] = {
                "residual": fits_dir / \
                    fits_template.format(mp=percent, t_int=t_int, itype="residual"),
                "dirty": fits_dir / \
                    fits_template.format(mp=percent, t_int=t_int, itype="dirty")
            }
        
    refreeze(params)

    solution_paths = [fluxes_paths[percent][t_int] for t_int in t_ints for percent in percents] \
        + [fits_paths[percent][t_int]["residual"] for t_int in t_ints for percent in percents] \
        + [fits_paths[percent][t_int]["dirty"] for t_int in t_ints for percent in percents]

    path = paths["fluxes"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not fluxes_dir.exists():
        os.mkdir(fluxes_dir)
    path = paths["fits"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not fits_dir.exists():
        os.mkdir(fits_dir)

    try:
        check_for_data(*solution_paths)
        logging.debug("Creating new solutions and images")
    except DataExistsError:
        logging.info("No deletion done")            
        return    

    total_runs = n_points * len(percents)
    model_path = paths["fluxes"]["true"]["files"][100]
    skymodel = load_data(model_path)
    model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    true_flux = model[Ix, Iy]
    gains_paths = paths["gains"]["quartical"]["files"]

    if progress:
        pbar.total = total_runs

    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    
    for percent in percents:
        for t_int in t_ints:
            gains_path = gains_paths[percent][t_int]
            flux_path = fluxes_paths[percent][t_int]
            residual_path = fits_paths[percent][t_int]["residual"]
            dirty_path = fits_paths[percent][t_int]["dirty"]
            vis_column = f"DATA_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_path, flux_path, 
                                residual_path, dirty_path)
            end = time.time()
            log_msg = f"QuartiCal on {percent}MP with "\
                    + f"`t-int={t_int}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                mse = mean_square_error(true_flux, recovered_flux)
                rms = np.sqrt(np.square(residual).mean())
                log_msg += f", with MSE={mse:.3g} and RMS={rms:.3g}"
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

    logging.info("Flux extractor complete")


def run_quartical_wsclean_imaging(params):
    logging.debug("Invoking function")
    paths = params["paths"]
    gains_dir = paths["gains"]["quartical"]
    percents = params["percents"]

    try:
        fits_paths = paths["fits"]["quartical"]
        fits_keys = fits_paths.keys()

        for percent in percents:
            if percent not in fits_keys:
                raise KeyError
            
        for key in fits_keys:
            if key not in percents:
                raise KeyError
        
        logging.debug("QuartiCal paths match")

    except:
        logging.info("Updating path data")
        data_dir = paths["data-dir"]
        with refreeze(paths) as file:
            if file.get("fits", True):
                file["fits"] = {
                    "dir" : data_dir / "fits"
                }
            fits_dir = file["fits"]
            os.makedirs(file["fits"]["dir"], exist_ok=True)

            if fits_dir.get("quartical", True):
                file["fits"]["quartical"] = {
                    "dir" : data_dir / "fits" / "quartical"
                }
            quartical_fits = fits_dir["quartical"]
            os.makedirs(fits_dir["quartical"]["dir"], exist_ok=True)

        logging.debug("Gains folders missing, created.")
    logging.info("Fetching imaging parameters")
    rms_noise = calculate_image_noise_rms(params)

    model_data = load_data(paths["fluxes"]["true"][100])
    n_pix = int(model_data["model"].shape[0])
    cell_asec = int(model_data["cell_asec"])
    t_ints = params["quartical"]["t-ints"]
    total_runs = len(t_ints) * len(percents)

    logging.info(f"Running `wsclean` on {total_runs} QuartiCal results.")
    for percent in percents:
        fits_paths = []
        for i, t_int in enumerate(t_ints):
            gains_path = gains_dir[percent][i]
            fits_path = quartical_fits["dir"] / \
                f"quartical-{percent}mp-t_int-{t_int}"
            
            calculate_imaging_columns(gains_path, params)

            wsclean_args = [
                "wsclean",
                "-j", str(params["n-cpu"]),
                "-abs-mem", "100",
                "-pol", "XX",
                "-size", str(n_pix), str(n_pix),
                "-scale", f"{cell_asec}asec",
                "-weight", f"briggs", "0.0",
                "-no-dirty", 
                "-niter", str(int(1e6)),
                "-threshold", str(2 * rms_noise),
                "-data-column", "CORRECTED_DATA",
                "-name", str(fits_path),
                str(paths["ms-path"])
            ]

            start = time()
            with open(os.devnull, 'w') as fp:
                subprocess.Popen(wsclean_args, 
                              stdout=fp, stderr=fp).wait()
            
            end = time()

            fits_paths.append(fits_path)

            logging.info(f"Finished `wsclean` with QuartiCal on {percent}MP with "\
                    + f"`t-int={t_int}`, {(end - start):.3g}s taken")
            
        logging.debug(f"Saving fits files paths for {percent}MP")    
        with refreeze(paths) as file:
            quartical_fits[percent] = fits_paths

def run_single_wsclean_imaging(gains_path, fits_name, params):
    logging.debug("Invoking function")
    paths = params["paths"]

    try:
        fits_dir = paths["fits"]["dir"]
        os.makedirs(fits_path, exist_ok=True)
        logging.debug("Fits directory exists")
    except:
        logging.info("Updating path data")
        data_dir = paths["data-dir"]
        with refreeze(paths) as file:
            if file.get("fits", True):
                file["fits"] = {
                    "dir" : data_dir / "fits"
                }
            os.makedirs(file["fits"]["dir"], exist_ok=True)

        logging.debug("Fits folders missing, created.")
    logging.info("Fetching imaging parameters")
    rms_noise = calculate_image_noise_rms(params)

    model_data = load_data(paths["fluxes"]["true"][100])
    n_pix = int(model_data["model"].shape[0])
    cell_asec = int(model_data["cell_asec"])
    calculate_imaging_columns(gains_path, params)
    fits_path = fits_dir / fits_name

    wsclean_args = [
        "wsclean",
        "-j", str(params["n-cpu"]),
        "-abs-mem", "100",
        "-pol", "XX",
        "-size", str(n_pix), str(n_pix),
        "-scale", f"{cell_asec}asec",
        "-weight", f"briggs", "0.0",
        "-no-dirty", 
        "-niter", str(int(1e6)),
        "-threshold", str(2 * rms_noise),
        "-data-column", "CORRECTED_DATA",
        "-name", str(fits_path),
        str(paths["ms-path"])
    ]

    start = time.time()
    print(" ".join(wsclean_args))
    # subprocess.Popen(wsclean_args).wait()
    
    end = time.time()
    logging.info(f"Finished `wsclean` on `{gains_path}` with {(end - start):.3g}s taken")