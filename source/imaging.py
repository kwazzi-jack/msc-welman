import numpy as np
from africanus.calibration.utils import corrupt_vis, correct_vis
import logging
from casacore.tables import table, maketabdesc, makearrcoldesc
from source.data import load_data
from source.parameters import refreeze
import os
import subprocess
from time import time

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

    start = time()
    print(" ".join(wsclean_args))
    # subprocess.Popen(wsclean_args).wait()
    
    end = time()
    logging.info(f"Finished `wsclean` on `{gains_path}` with {(end - start):.3g}s taken")