import numpy as np
from africanus.calibration.utils import corrupt_vis, correct_vis
from ducc0.wgridder import ms2dirty, dirty2ms
import logging
from casacore.tables import table, maketabdesc, makearrcoldesc
from source.data import load_data
from source.metrics import mean_square_error, root_mean_square_error
from source.other import progress_bar, check_for_data, DataExistsError
import os
import subprocess
import time
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime

def to4d(data):
    """From pfb.utils.misc"""

    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")
    
def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    """From pfb.utils.fits"""

    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))[:, :, ::-1]
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)


def set_wcs(cell_x, cell_y, nx, ny, radec, freq,
            unit='Jy/beam', GuassPar=None, unix_time=None):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in radians
    freq - frequencies in Hz
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    if np.size(freq) > 1:
        ref_freq = freq[0]
    else:
        ref_freq = freq
    w.wcs.crval = [radec[0]*180.0/np.pi, radec[1]*180.0/np.pi, ref_freq, 1]
    # LB - y axis treated differently because of stupid fits convention
    w.wcs.crpix = [1 + nx//2, ny//2, 1, 1]

    if np.size(freq) > 1:
        w.wcs.crval[2] = freq[0]
        df = freq[1]-freq[0]
        w.wcs.cdelt[2] = df
        fmean = np.mean(freq)
    else:
        if isinstance(freq, np.ndarray):
            fmean = freq[0]
        else:
            fmean = freq

    header = w.to_header()
    header['RESTFRQ'] = fmean
    header['ORIGIN'] = 'pfb-clean'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'
    if unix_time is not None:
        header['UTC_TIME'] = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')

    return header

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
                                model_column, model_file, 
                                fluxes_file, residual_file, 
                                dirty_file, corrected_file):
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
        model_vis = tb.getcol("MODEL_100MP")[..., 0].astype(np.complex128)

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
    gains = np.load(gains_file)["gains"]

    if gains.ndim == 2:
        gains = gains[..., None, None, None]

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
    logging.debug("Corrected-residual image created")

    residual_image = ms2dirty(uvw=UVW, freq=FREQ, ms=S * G.conj() * (vis - G * model_vis),
                    npix_x=nx, npix_y=ny,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    epsilon=tol, nthreads=n_cpu, do_wstacking=True)
    
    corrected_image = ms2dirty(uvw=UVW, freq=FREQ, ms=vis/G,
                    npix_x=nx, npix_y=ny, wgt=W,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    epsilon=tol, nthreads=n_cpu, do_wstacking=True)
    
    logging.debug("Recovered flux found, saved to file")
    residual_image /= W.sum()
    dirty_image /= W.sum()
    corrected_image /= W.sum()

    # Save flux per source to file
    with open(fluxes_file, "wb") as file:
        np.savez(file, 
                 flux=recovered_flux,
                 residual=residual_image,
                 corrected=corrected_image
                )
        
    image_to_fits(residual_image, cell_rad, residual_file, FREQ[0], RADEC)
    image_to_fits(corrected_image, cell_rad, corrected_file, FREQ[0], RADEC)
    image_to_fits(dirty_image, cell_rad, dirty_file, FREQ[0], RADEC)
    logging.debug("Fits files saved")


def run_kalcal_full_flux_extractor(params, progress=False, check_metric=False, overwrite=False):
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

    kalcal_params = params["kalcal-full"]
    status = kalcal_params["status"]
    sigma_fs = kalcal_params["sigma-fs"]
    n_points = kalcal_params["n-points"]

    if status == "DISABLED":
        logging.info("kalcal-full is disabled, do nothing")
        return
    
    logging.info("Retrieved line search sigma-fs parameters")
    logging.info("Updating path data")
    filter_fluxes_paths = paths["fluxes"]["kalcal-full"]["filter"]["files"]
    filter_fluxes_dir = paths["fluxes"]["kalcal-full"]["filter"]["dir"]
    filter_fluxes_template = paths["fluxes"]["kalcal-full"]["filter"]["template"]

    smoother_fluxes_paths = paths["fluxes"]["kalcal-full"]["smoother"]["files"]
    smoother_fluxes_dir = paths["fluxes"]["kalcal-full"]["smoother"]["dir"]
    smoother_fluxes_template = paths["fluxes"]["kalcal-full"]["smoother"]["template"]

    filter_fits_paths = paths["fits"]["kalcal-full"]["filter"]["files"]
    filter_fits_dir = paths["fits"]["kalcal-full"]["filter"]["dir"]
    filter_fits_template = paths["fits"]["kalcal-full"]["filter"]["template"]

    smoother_fits_paths = paths["fits"]["kalcal-full"]["filter"]["files"]
    smoother_fits_dir = paths["fits"]["kalcal-full"]["filter"]["dir"]
    smoother_fits_template = paths["fits"]["kalcal-full"]["filter"]["template"]

    for percent in percents:
        filter_fluxes_paths[percent] = {}
        smoother_fluxes_paths[percent] = {}
        filter_fits_paths[percent] = {}
        smoother_fits_paths[percent] = {}
        for sigma_f in sigma_fs:
            filter_fluxes_paths[percent][sigma_f] = filter_fluxes_dir / filter_fluxes_template.format(
                                           mp=percent, sigma_f=sigma_f)
            smoother_fluxes_paths[percent][sigma_f] = smoother_fluxes_dir / smoother_fluxes_template.format(
                                           mp=percent, sigma_f=sigma_f)            
            filter_fits_paths[percent][sigma_f] = {
                "residual": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="residual"),
                "dirty": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="dirty"),
                "corrected": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="corrected")
            }
            smoother_fits_paths[percent][sigma_f] = {
                "residual": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="residual"),
                "dirty": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="dirty"),
                "corrected": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="corrected")
            }

    solution_paths = [filter_fluxes_paths[percent][sigma_f] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["residual"] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["dirty"] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["corrected"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fluxes_paths[percent][sigma_f] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["residual"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["dirty"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["corrected"] for sigma_f in sigma_fs for percent in percents]

    path = paths["fluxes"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["fits"]["kalcal-full"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not filter_fluxes_dir.exists():
        os.mkdir(filter_fluxes_dir)
    if not smoother_fluxes_dir.exists():
        os.mkdir(smoother_fluxes_dir)

    path = paths["fits"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["fits"]["kalcal-full"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not filter_fits_dir.exists():
        os.mkdir(filter_fits_dir)
    if not smoother_fits_dir.exists():
        os.mkdir(smoother_fits_dir)

    params.save()
    
    if not overwrite:
        try:
            check_for_data(*solution_paths)
            logging.debug("Creating new solutions and images")
        except DataExistsError:
            logging.info("No deletion done")            
            return    
    else:
        logging.info("Overwriting previous solutions and images")

    total_runs = n_points * len(percents)
    model_path = paths["fluxes"]["true"]["files"][100]
    skymodel = load_data(model_path)
    model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    true_flux = model[Ix, Iy]
    filter_gains_paths = paths["gains"]["kalcal-full"]["filter"]["files"]
    smoother_gains_paths = paths["gains"]["kalcal-full"]["smoother"]["files"]

    if progress:
        pbar.total = 2 * total_runs

    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    
    for percent in percents:
        for sigma_f in sigma_fs:
            gains_path = filter_gains_paths[percent][sigma_f]
            flux_path = filter_fluxes_paths[percent][sigma_f]
            residual_path = filter_fits_paths[percent][sigma_f]["residual"]
            dirty_path = filter_fits_paths[percent][sigma_f]["dirty"]
            corrected_path = filter_fits_paths[percent][sigma_f]["corrected"]
            vis_column = f"DATA_100MP"
            model_column = f"MODEL_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_column, model_path, flux_path, 
                                residual_path, dirty_path, corrected_path)
            end = time.time()
            log_msg = f"kalcal-full-filter on {percent}MP with "\
                    + f"`sigma-f={sigma_f:.3g}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                corrected = results["corrected"]
                mse = mean_square_error(true_flux, recovered_flux)
                res_rms = np.sqrt(np.square(residual).mean())
                cor_rms = np.sqrt(np.square(corrected).mean())
                log_msg += f", with Flux MSE={mse:.3g}, "
                log_msg += f"Residual RMS={res_rms:.3g}, and "
                log_msg += f"Corrected RMS={cor_rms:.3g}."
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()
            
            gains_path = smoother_gains_paths[percent][sigma_f]
            flux_path = smoother_fluxes_paths[percent][sigma_f]
            residual_path = smoother_fits_paths[percent][sigma_f]["residual"]
            dirty_path = smoother_fits_paths[percent][sigma_f]["dirty"]
            corrected_path = smoother_fits_paths[percent][sigma_f]["corrected"]
            vis_column = f"DATA_100MP"
            model_column = f"MODEL_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_column, model_path, flux_path, 
                                residual_path, dirty_path, corrected_path)
            end = time.time()
            log_msg = f"kalcal-full-smoother on {percent}MP with "\
                    + f"`sigma-f={sigma_f:.3g}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                corrected = results["corrected"]
                mse = mean_square_error(true_flux, recovered_flux)
                res_rms = np.sqrt(np.square(residual).mean())
                cor_rms = np.sqrt(np.square(corrected).mean())
                log_msg += f", with Flux MSE={mse:.3g}, "
                log_msg += f"Residual RMS={res_rms:.3g}, and "
                log_msg += f"Corrected RMS={cor_rms:.3g}."
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

    logging.info("Flux extractor complete")


def run_kalcal_diag_flux_extractor(params, progress=False, check_metric=False, overwrite=False):
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

    kalcal_params = params["kalcal-diag"]
    status = kalcal_params["status"]
    sigma_fs = kalcal_params["sigma-fs"]
    n_points = kalcal_params["n-points"]

    if status == "DISABLED":
        logging.info("kalcal-diag is disabled, do nothing")
        return
    
    logging.info("Retrieved line search sigma-fs parameters")
    logging.info("Updating path data")
    filter_fluxes_paths = paths["fluxes"]["kalcal-diag"]["filter"]["files"]
    filter_fluxes_dir = paths["fluxes"]["kalcal-diag"]["filter"]["dir"]
    filter_fluxes_template = paths["fluxes"]["kalcal-diag"]["filter"]["template"]

    smoother_fluxes_paths = paths["fluxes"]["kalcal-diag"]["smoother"]["files"]
    smoother_fluxes_dir = paths["fluxes"]["kalcal-diag"]["smoother"]["dir"]
    smoother_fluxes_template = paths["fluxes"]["kalcal-diag"]["smoother"]["template"]

    filter_fits_paths = paths["fits"]["kalcal-diag"]["filter"]["files"]
    filter_fits_dir = paths["fits"]["kalcal-diag"]["filter"]["dir"]
    filter_fits_template = paths["fits"]["kalcal-diag"]["filter"]["template"]

    smoother_fits_paths = paths["fits"]["kalcal-diag"]["filter"]["files"]
    smoother_fits_dir = paths["fits"]["kalcal-diag"]["filter"]["dir"]
    smoother_fits_template = paths["fits"]["kalcal-diag"]["filter"]["template"]

    for percent in percents:
        filter_fluxes_paths[percent] = {}
        smoother_fluxes_paths[percent] = {}
        filter_fits_paths[percent] = {}
        smoother_fits_paths[percent] = {}
        for sigma_f in sigma_fs:
            filter_fluxes_paths[percent][sigma_f] = filter_fluxes_dir / filter_fluxes_template.format(
                                           mp=percent, sigma_f=sigma_f)
            smoother_fluxes_paths[percent][sigma_f] = smoother_fluxes_dir / smoother_fluxes_template.format(
                                           mp=percent, sigma_f=sigma_f)            
            filter_fits_paths[percent][sigma_f] = {
                "residual": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="residual"),
                "dirty": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="dirty"),
                "corrected": filter_fits_dir / \
                    filter_fits_template.format(mp=percent, sigma_f=sigma_f, itype="corrected")
            }
            smoother_fits_paths[percent][sigma_f] = {
                "residual": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="residual"),
                "dirty": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="dirty"),
                "corrected": smoother_fits_dir / \
                    smoother_fits_template.format(mp=percent, sigma_f=sigma_f, itype="corrected")
            }

    solution_paths = [filter_fluxes_paths[percent][sigma_f] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["residual"] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["dirty"] for sigma_f in sigma_fs for percent in percents] \
        + [filter_fits_paths[percent][sigma_f]["corrected"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fluxes_paths[percent][sigma_f] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["residual"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["dirty"] for sigma_f in sigma_fs for percent in percents] \
        + [smoother_fits_paths[percent][sigma_f]["corrected"] for sigma_f in sigma_fs for percent in percents]

    path = paths["fluxes"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["fits"]["kalcal-diag"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not filter_fluxes_dir.exists():
        os.mkdir(filter_fluxes_dir)
    if not smoother_fluxes_dir.exists():
        os.mkdir(smoother_fluxes_dir)

    path = paths["fits"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["fits"]["kalcal-diag"]["dir"]
    if not path.exists():
        os.mkdir(path)
    if not filter_fits_dir.exists():
        os.mkdir(filter_fits_dir)
    if not smoother_fits_dir.exists():
        os.mkdir(smoother_fits_dir)

    params.save()
    
    if not overwrite:
        try:
            check_for_data(*solution_paths)
            logging.debug("Creating new solutions and images")
        except DataExistsError:
            logging.info("No deletion done")            
            return    
    else:
        logging.info("Overwriting previous solutions and images")

    total_runs = n_points * len(percents)
    model_path = paths["fluxes"]["true"]["files"][100]
    skymodel = load_data(model_path)
    model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    true_flux = model[Ix, Iy]
    filter_gains_paths = paths["gains"]["kalcal-diag"]["filter"]["files"]
    smoother_gains_paths = paths["gains"]["kalcal-diag"]["smoother"]["files"]

    if progress:
        pbar.total = 2 * total_runs

    logging.info(f"Running line-search on {n_points} points " \
                + f"({total_runs} runs)")
    
    for percent in percents:
        for sigma_f in sigma_fs:
            gains_path = filter_gains_paths[percent][sigma_f]
            flux_path = filter_fluxes_paths[percent][sigma_f]
            residual_path = filter_fits_paths[percent][sigma_f]["residual"]
            dirty_path = filter_fits_paths[percent][sigma_f]["dirty"]
            corrected_path = filter_fits_paths[percent][sigma_f]["corrected"]
            vis_column = f"DATA_100MP"
            model_column = f"MODEL_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_column, model_path, flux_path, 
                                residual_path, dirty_path, corrected_path)
            end = time.time()
            log_msg = f"kalcal-diag-filter on {percent}MP with "\
                    + f"`sigma-f={sigma_f:.3g}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                corrected = results["corrected"]
                mse = mean_square_error(true_flux, recovered_flux)
                res_rms = np.sqrt(np.square(residual).mean())
                cor_rms = np.sqrt(np.square(corrected).mean())
                log_msg += f", with Flux MSE={mse:.3g}, "
                log_msg += f"Residual RMS={res_rms:.3g}, and "
                log_msg += f"Corrected RMS={cor_rms:.3g}."
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()
            
            gains_path = smoother_gains_paths[percent][sigma_f]
            flux_path = smoother_fluxes_paths[percent][sigma_f]
            residual_path = smoother_fits_paths[percent][sigma_f]["residual"]
            dirty_path = smoother_fits_paths[percent][sigma_f]["dirty"]
            corrected_path = smoother_fits_paths[percent][sigma_f]["corrected"]
            vis_column = f"DATA_100MP"
            model_column = f"MODEL_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_column, model_path, flux_path, 
                                residual_path, dirty_path, corrected_path)
            end = time.time()
            log_msg = f"kalcal-diag-smoother on {percent}MP with "\
                    + f"`sigma-f={sigma_f:.3g}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                corrected = results["corrected"]
                mse = mean_square_error(true_flux, recovered_flux)
                res_rms = np.sqrt(np.square(residual).mean())
                cor_rms = np.sqrt(np.square(corrected).mean())
                log_msg += f", with Flux MSE={mse:.3g}, "
                log_msg += f"Residual RMS={res_rms:.3g}, and "
                log_msg += f"Corrected RMS={cor_rms:.3g}."
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

    logging.info("Flux extractor complete")

def run_quartical_flux_extractor(params, progress=False, check_metric=False, overwrite=False):
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
    n_points = quart_params["n-points"]

    if status == "DISABLED":
        logging.info("QuartiCal is disabled, do nothing")
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
                    fits_template.format(mp=percent, t_int=t_int, itype="dirty"),
                "corrected": fits_dir / \
                    fits_template.format(mp=percent, t_int=t_int, itype="corrected")
            }

    solution_paths = [fluxes_paths[percent][t_int] for t_int in t_ints for percent in percents] \
        + [fits_paths[percent][t_int]["residual"] for t_int in t_ints for percent in percents] \
        + [fits_paths[percent][t_int]["dirty"] for t_int in t_ints for percent in percents] \
        + [fits_paths[percent][t_int]["corrected"] for t_int in t_ints for percent in percents]

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

    params.save()
    
    if not overwrite:
        try:
            check_for_data(*solution_paths)
            logging.debug("Creating new solutions")
        except DataExistsError:
            logging.info("No deletion done")
            return
    else:
        logging.debug("Overwriting previous solutions")
        
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
            corrected_path = fits_paths[percent][t_int]["corrected"]
            vis_column = f"DATA_100MP"
            model_column = f"MODEL_{percent}MP"
            
            start = time.time()
            calculate_recovered_flux(params, gains_path, vis_column, 
                                model_column, model_path, flux_path, 
                                residual_path, dirty_path, corrected_path)
            end = time.time()
            log_msg = f"QuartiCal on {percent}MP with "\
                    + f"`t-int={t_int}`, {(end - start):.3g}s taken"
                         
            if check_metric:
                results = load_data(flux_path)
                recovered_flux = results["flux"]
                residual = results["residual"]
                corrected = results["corrected"]
                mse = mean_square_error(true_flux, recovered_flux)
                res_rms = np.sqrt(np.square(residual).mean())
                cor_rms = np.sqrt(np.square(corrected).mean())
                log_msg += f", with Flux MSE={mse:.3g}, "
                log_msg += f"Residual RMS={res_rms:.3g}, and "
                log_msg += f"Corrected RMS={cor_rms:.3g}."
            
            logging.info(log_msg)
            if progress:
                pbar.update(1)
                pbar.refresh()

    logging.info("Flux extractor complete")

def run_true_flux_extractor(params, overwrite=False):
    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    percents = params["percents"]
    fits_paths = paths["fits"]["true"]["files"]
    fits_dir = paths["fits"]["true"]["dir"]
    fits_template = paths["fits"]["true"]["template"]
    flux_paths = {percent: paths["fluxes"]["true"]["dir"] / f"true-flux-{percent}mp.npz" for percent in percents}
    paths["fluxes"]["true"]["solved"] = flux_paths
    fits_paths["residual"] = {}
    fits_paths["dirty"] = {}
    fits_paths["corrected"] = {}

    for percent in percents:
        fits_paths["residual"][percent] = fits_dir / fits_template.format(itype="residual", mp=percent)
        fits_paths["dirty"][percent] = fits_dir / fits_template.format(itype="dirty", mp=percent)
        fits_paths["corrected"][percent] = fits_dir / fits_template.format(itype=f"corrected", mp=percent)

    solution_paths = [fits_paths[itype][percent] for percent in percents for itype in fits_paths.keys()]

    if not fits_dir.exists():
        os.mkdir(fits_dir)
    params.save()
    
    if not overwrite:
        try:
            check_for_data(*solution_paths)
            logging.debug("Creating new solutions and images")
        except DataExistsError:
            logging.info("No deletion done")            
            return    
    else:
        logging.info("Overwriting previous solutions and images")

    model_path = paths["fluxes"]["true"]["files"][100]
    skymodel = load_data(model_path)
    gains_path = paths["gains"]["true"]["files"]

    for percent in percents:
        flux_path = flux_paths[percent]
        residual_path = fits_paths["residual"][percent]
        dirty_path = fits_paths["dirty"][percent]
        corrected_path = fits_paths["corrected"][percent]
        vis_column = f"DATA_100MP"
        model_column = f"MODEL_{percent}MP"
        
        calculate_recovered_flux(params, gains_path, vis_column, 
                            model_column, model_path, flux_path, 
                            residual_path, dirty_path, corrected_path)
        
        logging.info(f"Completed true images on {percent}MP")
    logging.info("Flux extractor complete")

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