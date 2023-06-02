from source.parameters import Settings, YamlDict, refreeze
from pathlib import Path
from ipywidgets import Output
from IPython.display import display, clear_output
from subprocess import Popen
from source.other import check_for_data, DataExistsError
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
from africanus.coordinates import radec_to_lm
from casacore.tables import table, makearrcoldesc, maketabdesc
import numpy as np
import logging
import random
from ducc0.wgridder import dirty2ms
from africanus.constants import c as light_speed
from africanus.calibration.utils import corrupt_vis, correct_vis
from ducc0.fft import good_size
import os

def measurement_set_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating measurement set options config")
    paths = params["paths"]
    name = "ms-options.yml"
    options_dir = paths["options"]["dir"]
    with refreeze(params):
        paths["options"]["ms"] = options_dir / name
    logging.debug(f"Measurement set config at: `{options_dir / name}`")

    settings = Settings(
        name=name,
        header="Measurement Set",
        description="""Settings to generate an empty measurement set for the simulation.
        It leverages the software `simms` to create a suitable database for this experiment.""",
        directory=options_dir,
        immutable_path=True,
    )
    logging.debug("Measurement Set Settings object created")

    settings["ms-name"] = (
        "Measurement Set Name",
        "Name and path of measurement set. If empty, then telescope name.",
        "meerkat.ms",
    )
    logging.debug("Setting added: `ms-name`")

    settings["telescope"] = (
        "Radio Telescope Array",
        "Choose the radiointerferometer to use for the experiment. Default: MeerKAT.",
        ["MeerKAT", "KAT-7", "VLA-A", "VLA-B"],
    )
    logging.debug("Setting added: `telescope`")

    settings["ra"] = (
        "Right Ascension (HH:MM:SS)",
        "Designated right ascension of array. Not checked against array position.",
        "11h50m15s",
    )
    logging.debug("Setting added: `ra`")

    settings["dec"] = (
        "Declination (DD:MM:SS)",
        "Designated declination of array. Not checked against array position.",
        "-30d27m43s",
    )
    logging.debug("Setting added: `dec`")

    settings["synth-time"] = (
        "Synthesis Time (hours)",
        "Length of the observation in time.",
        2,
    )
    logging.debug("Setting added: `synth-time`")

    settings["d-time"] = (
        "Integration Time (seconds)",
        "Rate at which observations are taken.",
        10,
    )
    logging.debug("Setting added: `d-time`")

    settings["freq-0"] = (
        "Initial Frequency (Hz)",
        "The main viewing band for experiment.",
        "1GHz",
    )
    logging.debug("Setting added: `d-time`")

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("Measurement set settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("Measurement set settings set to default and saved")

    logging.debug("Measurement set settings complete and returning")
    return settings


def create_empty_measurement_set(ms_options, params, verbose=False):
    logging.debug("Invoking function")

    logging.debug("Retrieve measurement set options")
    paths = params["paths"]
    telescope = str(ms_options["telescope"]).lower()
    ms_name = ms_options["ms-name"] if len(ms_options["ms-name"]) else f"{telescope}.ms"
    ra, dec = ms_options["ra", "dec"]
    st, dt = ms_options["synth-time", "d-time"]
    freq_0 = ms_options["freq-0"]

    data_dir = paths["data"]["dir"]
    try:
        ms_path = paths["data"]["ms"]
        logging.debug("Using existing path")
    except:
        logging.debug("Create new path")
        ms_path = data_dir / ms_name

    logging.debug(f"Path to measurement set created: `{ms_path}`")
    logging.debug(f"Adjusting frequency input")

    if "Hz" in freq_0:
        freq_0 = freq_0.replace(" ", "")
    else:
        freq_0 += "Hz"

    logging.info(f"Name/Path: `{ms_path}`")
    logging.info(f"Telescope: {telescope}")
    logging.info(f"Right Ascension: {ra}")
    logging.info(f"Declination: {dec}")
    logging.info(f"Synthesis-Time: {st} hr")
    logging.info(f"Integration-Time: {dt} s")
    logging.info(f"Initial Frequency: {freq_0}")

    try:
        logging.debug("Checking if measurement set exists")
        check_for_data(ms_path)
        logging.debug("Creating `simms` arguments.")
        simms_args = [
            "simms",
            "--type",
            "ascii",
            "--coord-sys",
            "itrf",
            "--tel",
            telescope,
            "--name",
            str(ms_path),
            "--ra",
            str(ra),
            "--dec",
            str(dec),
            "--synthesis-time",
            str(st),
            "--dtime",
            str(dt),
            "--nchan",
            "1",
            "--freq0",
            str(freq_0),
            "--dfreq",
            "1MHz",
            "--pol",
            "XX",
            "--nolog",
        ]

        out = Output(layout={"border": "1px solid black"})
        if verbose:
            logging.debug(f"Verbose `simms` output")
            display(out)

        logging.info("Create measurement set with `simms`")
        with out:
            Popen(simms_args).wait()

        logging.info("Updating path and parameter data")
        with refreeze(params):
            params["telescope"] = telescope
            params["ra"] = ra
            params["dec"] = dec
            params["st"] = st
            params["dt"] = dt
            params["freq_0"] = freq_0
            paths["data"]["ms"] = ms_path

        logging.info(f"New measurement set created at `{ms_path}`")
    except:
        logging.info(f"Keep original measurement set at `{ms_path}`")
        logging.info("Updating path and parameter data")
        with refreeze(params):
            params["ra"] = ra
            params["dec"] = dec
            params["st"] = st
            params["dt"] = dt
            params["freq_0"] = freq_0
            paths["data"]["ms"] = ms_path


def gains_options(params, no_gui=False):
    logging.debug("Invoking function")

    logging.debug("Creating gains options config")
    paths = params["paths"]
    name = "gains-config.yml"
    options_dir = paths["options"]["dir"]
    with refreeze(params):
        paths["options"]["gains"] = options_dir / name
    logging.debug(f"Measurement set config at: `{options_dir / name}`")
    
    settings = Settings(
        name=name,
        header="Gains Signal",
        description="""
        These settings determine the configuration of the true
        underlying gains signal that will corrupt the true
        visibilities. It consists of two real Gaussian processes
        created for the amplitude and phase components of the
        gains. Each Gaussian process is given a mean and the
        squared exponential kernel with a certain length scale. Once
        generated, they are combined to create the final complex
        gains signal and saved to file.
        """,
        directory=options_dir,
        immutable_path=True,
    )
    logging.debug("Gains options object created")

    settings["amp-length"] = (
        "Amplitude length scale",
        "Determines rate of amplitude variation over time.",
        100.0,
    )
    logging.debug("Setting added: `amp-length`")

    settings["amp-var"] = (
        "Variation of amplitude",
        "Standard deviation of the amplitude signal over time.",
        0.2,
    )
    logging.debug("Setting added: `amp-var`")

    settings["amp-mean"] = (
        "Mean of phase",
        "Centre the amplitude-signal around a mean over time.",
        1.0,
    )
    logging.debug("Setting added: `amp-mean`")

    settings["phase-length"] = (
        "Phase length scale",
        "Determines rate of variation over time.",
        10.0,
    )
    logging.debug("Setting added: `phase-length`")

    settings["phase-var"] = (
        "Variation of phase",
        "Standard deviation of the phase signal over time.",
        0.2,
    )
    logging.debug("Setting added: `phase-var`")

    settings["phase-mean"] = (
        "Mean of phase",
        "Centre the phase-signal around a mean over time.",
        0.0,
    )
    logging.debug("Setting added: `phase-mean`")

    settings["type"] = (
        "Type of gains",
        "Isolate the gains type for the true gains.",
        ["Full-complex", "Amplitude-only", "Phase-only", "Unity"],
    )
    logging.debug("Setting added: `type`")

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(name)
        logging.debug("Gains options config loaded from file")
    except:
        settings.to_yaml(name)
        logging.debug("Gains options config set to default and saved")

    logging.debug("Gains options config complete and returning")
    return settings


def __create_gains(axes, scales, stds, shape, mean=0.0):
    """Produce complex-gains based on the dimensions given."""
    # Get axes values
    t, nu, s = axes

    # Get length scales
    lt, lnu, ls = scales

    # Get variations
    st, snu, ss = stds

    # Gains dimensions
    n_time, n_ant, n_chan, n_dir, n_corr = shape

    # Scale down domain
    t = np.arange(t.size)
    nu = nu / nu.max() if nu.max() != 0 else nu
    s = s / s.max() if s.max() != 0 else s

    # Make prior covariace matrices
    Kt = expsq(t, t, st, lt)
    Knu = expsq(nu, nu, snu, lnu)
    Ks = expsq(s, s, ss, ls)

    # Stack and get cholesky factors
    K = np.array((Kt, Knu, Ks), dtype=object)
    L = kt.kron_cholesky(K)

    # Simulate independent gain per antenna and direction
    gains = np.zeros((n_time, n_ant, n_chan, n_dir, n_corr), dtype=np.float64)

    for p in range(n_ant):
        for c in range(n_corr):
            # Generate random complex vector
            xi = np.random.randn(n_time, n_chan, n_dir)

            # Apply to field
            gains[:, p, :, :, c] = (
                kt.kron_matvec(L, xi).reshape(n_time, n_chan, n_dir) + mean
            )

    # Return complex-gains
    return gains


def create_gains_signal(gs_options, params):
    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    logging.info("Updating path data")
    true_path = paths["gains"]["true"].get("files", False)
    if true_path and isinstance(true_path, Path):
        true_path = paths["gains"]["true"]["files"]
        logging.debug("True gains path exists, do nothing")
    else:
        true_path = paths["gains"]["true"]["dir"] / paths["gains"]["true"]["template"]
        logging.debug(f"True gains path does not exist, create new one: {true_path}")

    logging.debug("Check if directories exist")
    path = paths["gains"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["gains"]["true"]["dir"]
    if not path.exists():
        os.mkdir(path)

    try:
        check_for_data(true_path)
        logging.info("Creating new true gains")
    except DataExistsError:
        logging.info("Using existing gains")
        logging.info("Updating path and parameter data")
        with refreeze(params):
            paths["gains"]["true"]["files"] = true_path
        return
    
    logging.info(f"Retrieving information from `{paths['data']['ms']}`")
    with table(str(paths["data"]["ms"]), ack=False) as tb:
        TIME = tb.getcol("TIME")
        ANT1 = tb.getcol("ANTENNA1")
        ANT2 = tb.getcol("ANTENNA2")

    # Open field subtable and retrieve data
    with table(str(paths["data"]["ms"]) + "::FIELD", ack=False) as tb:
        PHASE_DIR = tb.getcol("PHASE_DIR").astype(np.float64)
        
    # Open spectral subtable and retrieve data
    with table(str(paths["data"]["ms"]) + "::SPECTRAL_WINDOW", ack=False) as tb:
        FREQ = tb.getcol("CHAN_FREQ").flatten()[0].astype(np.float64)
        
    # Time indices and axis size
    logging.info("Calculating dimensions and axes")
    _, tbin_indices = np.unique(TIME, return_index=True)
    n_time = len(tbin_indices)

    # Number of antennas
    n_ant = np.max((ANT1.max(), ANT2.max())) + 1

    # Other dimension sizes
    n_chan = n_dir = n_corr = 1

    # Final gains shape
    shape = (n_time, n_ant, n_chan, n_dir, n_corr)

    logging.debug("Updating dimension information")
    with refreeze(params):
        params["n-ant"] = n_ant
        params["n-time"] = n_time
        params["n-chan"] = n_chan
        params["n-dir"] = n_dir
        params["n-corr"] = n_corr

    # Create lm-array
    lm = np.array(radec_to_lm(PHASE_DIR.reshape(1, -1)))

    # Axes ranges
    axes = (tbin_indices, FREQ, lm)

    # Amplitude Parameters
    amp_scales = (gs_options["amp-length"], 1.0, 1.0)
    amp_stds = (gs_options["amp-var"], 1.0, 1.0)
    logging.debug("Amplitude information set")

    # Phase Parameters
    phase_scales = (gs_options["phase-length"], 1.0, 1.0)
    phase_stds = (gs_options["phase-var"], 1.0, 1.0)
    logging.debug("Phase information set")
    
    # Make complex-gains using above
    if gs_options["type"].lower() == "full-complex":
        logging.info("Generating full-complex gains")
        amp_gains = __create_gains(
            axes, amp_scales, amp_stds, shape, mean=gs_options["amp-mean"]
        )
        logging.debug("Amplitude signal generated")
        phase_gains = __create_gains(
            axes, phase_scales, phase_stds, shape, mean=gs_options["phase-mean"]
        )
        logging.debug("Phase signal generated")
        complex_gains = amp_gains * np.exp(1.0j * phase_gains)
        logging.debug("Final complex gains created from amplitude and phase")
    elif gs_options["type"].lower() == "amplitude-only":
        logging.info("Generating amplitude-only gains")
        amp_gains = __create_gains(
            axes, amp_scales, amp_stds, shape, mean=gs_options["amp-mean"]
        )
        logging.debug("Amplitude signal generated")
        phase_gains = np.ones(shape)
        logging.debug("Phase is unity")
        complex_gains = amp_gains * np.exp(1.0j * phase_gains)
        logging.debug("Final complex gains created from amplitude and phase")
    elif gs_options["type"].lower() == "phase-only":
        logging.info("Generating phase-only gains")
        amp_gains = np.ones(shape)
        logging.debug("Amplitude is unity")
        phase_gains = __create_gains(
            axes, phase_scales, phase_stds, shape, mean=gs_options["phase-mean"]
        )
        logging.debug("Phase signal generated")
        complex_gains = amp_gains * np.exp(1.0j * phase_gains)
        logging.debug("Final complex gains created from amplitude and phase")
    else:
        logging.info("Generating unity gains")
        amp_gains = np.ones(shape)
        logging.debug("Amplitude is unity")
        phase_gains = np.ones(shape)
        logging.debug("Phase is unity")
        complex_gains = np.ones(shape)
        logging.debug("Final complex gains created from amplitude and phase")
    
    logging.info(f"Codex shape: (N_ant, N_time, N_chan, N_dir, N_corr) = {shape}")
    logging.info("Saving gains to file")
    with open(true_path, "wb") as file:
        np.savez(file, 
            gains=complex_gains, 
            amp_gains=amp_gains, 
            phase_gains=phase_gains
            )

    logging.info(f"New simulated gains at `{true_path}`")
    logging.info("Updating path and parameter data")
    with refreeze(params):
            paths["gains"]["true"]["files"] = true_path

def load_data(path):
    logging.debug("Invoking function")
    logging.debug(f"Loading data from `{path}`")
    item = np.load(path)
    items = dict()
    for key in item.files:
        value = item[key]
        logging.debug(f"Found `array={key}` with `shape={value.shape}` and `dtype={value.dtype}`")

        if value.ndim == 5:
            logging.debug("Codex format, reducing to time and antenna")
            value = value[:, :, 0, 0, 0]
        elif value.ndim == 6:
            logging.debug("kalcal format, reducing to time and antenna")
            value = value[:, :, 0, 0, 0, 0]
        else:
            logging.debug("Other format, do nothing")

        logging.debug("Saving to dictionary")       
        items[key] = value

    logging.debug("Returning loaded terms")
    return items

def visibility_options(params, no_gui=False):
    logging.debug("Invoking function")

    logging.debug("Creating visibility options config")
    paths = params["paths"]

    name = "visibility-config.yml"
    options_dir = paths["options"]["dir"]
    with refreeze(params):
        paths["options"]["vis"] = options_dir / name    
    logging.debug(f"Visibility options config at: `{options_dir / name}`")
    
    settings = Settings(
        name=name,
        header="Visibility Settings",
        description="""
        Options that control the output discretized image of the
        sky to be used within the simulation and the creation of
        visibility data.
        """,
        directory=options_dir,
        immutable_path=True,
    )
    logging.debug("Visibility options object created")

    settings["percents"] = (
        "Modelled Flux Percentages",
        "Comma separated list of modelled flux percentage to use.",
        "100, 75, 50, 25",
    )
    logging.debug("Setting added: `percents`")

    settings["sigma-n"] = (
        "Measurement noise",
        "Standard deviation of the white noise present in measurements.",
        2.0,
    )
    logging.debug("Setting added: `sigma-n`")

    settings["fov"] = (
        "Field of View (deg)",
        "The capture area of the array in degrees.",
        0.25,
    )
    logging.debug("Setting added: `fov`")

    settings["peak-flux"] = (
        "Peak Flux",
        "The flux of the brightest source(s) in the FOV.",
        0.5,
    )
    logging.debug("Setting added: `peak-flux`")

    settings["min-flux"] = (
        "Minimum flux",
        "The minimum amount of flux all sources should be above.",
        0.01,
    )
    logging.debug("Setting added: `min-flux`")

    settings["dist"] = (
        "Flux Distribution",
        "The distribution type to generate the flux from.",
        ["Pareto", "Linear", "Constant", "Custom"],
    )
    logging.debug("Setting added: `dist`")

    settings["alpha"] = (
        "Alpha Value",
        "Corresponding parameter value for the selected distribution.",
        4.0,
    )
    logging.debug("Setting added: `alpha`")

    settings["n-src"] = (
        "Number of Sources",
        "How many sources to place in the image.",
        100,
    )
    logging.debug("Setting added: `n-src`")

    settings["buffer"] = (
        "Buffer between sources",
        "The amount of pixels to place between sources.",
        4,
    )
    logging.debug("Setting added: `buffer`")

    settings["layout"] = (
        "Source Layout",
        "Determines the position layout of the sources in the image.",
        ["Random", "Grid", "Custom"],
    )
    logging.debug("Setting added: `layout`")

    settings["sampling"] = (
        "Under/Over Nyquist Sampling",
        "Scale factor change the cell-size to under or over sample.",
        0.9,
    )
    logging.debug("Setting added: `sampling`")

    settings["tol"] = (
        "Tolerance",
        "Tolerance to use for the image-to-visibility step.",
        0.0000001,
    )
    logging.debug("Setting added: `sampling`")

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(name)
        logging.debug("Visibility options config loaded from file")
    except:
        settings.to_yaml(name)
        logging.debug("Visibility options config set to default and saved")

    logging.debug("Visibility options config complete and returning")
    return settings

def __grid_to_image(u, v, buffer):
    u_in = np.random.randint(buffer, 2 * buffer)
    v_in = np.random.randint(buffer, 2 * buffer)
    
    u_out = 3 * buffer * u + u_in
    v_out = 3 * buffer * v + v_in
    
    return [u_out, v_out]

def create_skymodels(vis_options, params, 
                     position_function=None, flux_function=None):
    
    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
        random.seed(params["seed"])
    else:
        logging.info(f"No seed set")
    
    percents = list(map(int, vis_options["percents"].replace(" ", "").split(",")))

    logging.info("Updating path data")
    true_paths = paths["fluxes"]["true"]["files"]
    try:
        if true_paths:
            true_keys = true_paths.keys()
            for percent in percents:
                if percent not in true_keys:
                    raise KeyError
            
            for percent in true_keys:
                if percent not in percents:
                    raise KeyError
                
            logging.debug("True fluxes paths exists, do nothing")
        else:
            raise KeyError
    except KeyError:
        logging.debug("True fluxes paths do not exist, create new ones")
        template = paths["fluxes"]["true"]["template"]
        base_dir = paths["fluxes"]["true"]["dir"]
        with refreeze(params):
            for percent in percents:
                true_paths[percent] = base_dir / template.format(mp=percent)

    logging.debug("Check if directories exist")
    path = paths["fluxes"]["dir"]
    if not path.exists():
        os.mkdir(path)
    path = paths["fluxes"]["true"]["dir"]
    if not path.exists():
        os.mkdir(path)

    try:
        check_for_data(*true_paths.values())
        logging.info("Creating new true fluxes")
    except DataExistsError:
        logging.info("Using existing true fluxes")
        return

    # Model Parameters
    logging.info("Fetching skymodel information")
    peak_flux = vis_options["peak-flux"]
    n_src = vis_options["n-src"]
    alpha = vis_options["alpha"]
    fov = vis_options["fov"]
    min_flux = vis_options["min-flux"]
    buffer = vis_options["buffer"]
    sampling = vis_options["sampling"]
    dist = vis_options["dist"]
    layout = vis_options["layout"]

    logging.info("Updating parameter information")
    with refreeze(params):
        params["percents"] = percents
        params["peak-flux"] = peak_flux
        params["n-src"] = n_src
        params["alpha"] = alpha
        params["fov"] = fov
        params["min-flux"] = min_flux
        params["buffer"] = buffer
        params["sampling"] = sampling
        params["dist"] = dist
        params["layout"] = layout

    logging.debug("Calculating cell-size and n-pix")
    with table(str(paths["data"]["ms"]), ack=False) as tb:
        UVW = tb.getcol("UVW")
        
    with table(str(paths["data"]["ms"]) + "::SPECTRAL_WINDOW", ack=False) as tb:
        FREQ = tb.getcol("CHAN_FREQ").flatten()[0]

    uv_max = np.abs(UVW[:, 0:2]).max()
    cell_rad = sampling/(2 * uv_max * FREQ/light_speed)
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg * 3600

    n_pix = good_size(int(np.ceil(fov/cell_deg)))
    while n_pix % 2:
        n_pix += 1
        n_pix = good_size(n_pix)        
    nx = ny = n_pix

    if dist.lower() == "pareto":
        logging.debug("Pareto distribution used for flux")
        flux = -np.sort(-np.random.pareto(alpha, n_src))
        scale = flux.max()/peak_flux
        flux /= scale
        mean_adjust = min_flux/peak_flux
        flux = flux * (1.0 - mean_adjust) + peak_flux * mean_adjust
    elif dist.lower() == "linear":
        logging.debug("Linear ramp used for flux")
        flux = peak_flux / (alpha * np.arange(n_src) + 1)
        mean_adjust = min_flux/peak_flux
        flux = flux * (1.0 - mean_adjust) + peak_flux * mean_adjust
    elif dist.lower() == "constant":
        logging.debug("Constant used for flux")
        flux = peak_flux * np.ones(n_src)
    else:
        logging.debug("Custom function used for flux")
        flux = flux_function(peak_flux, min_flux, alpha, n_src)
    
    logging.info("Determining flux and layout information")
    cumsum = np.cumsum(flux)
    total_percent = 100 * cumsum/np.sum(flux)
    models = np.zeros((len(percents), nx, ny), dtype=np.float64)

    if layout.lower() == "random":
        logging.debug("Layout set to random")
        Ix, Iy = [], []

        n_boxes = n_pix // (3 * buffer)
        bag = [(u, v) for u in range(1, n_boxes - 1) 
                    for v in range(1, n_boxes - 1)]

        coords = np.array([__grid_to_image(u, v, buffer) 
                           for u, v in random.sample(bag, n_src)])
        Ix = coords[:, 0]
        Iy = coords[:, 1]
        
        distances = []
        for x, y in coords:
            for u, v in coords:
                if u == x and y == v:
                    continue
                
                distances.append(np.sqrt((u - x)**2 + (v - y)**2))

        distances = np.array(distances)
        if np.where(distances < buffer, 1.0, 0.0).any():
            logging.error(f"Sources within buffer reach of {buffer}")
            raise ValueError("Some sources are within buffer reach")
    else:
        logging.error("Other layouts not yet completed.")
        raise NotImplementedError("Other layouts not yet implemented.")
    
    logging.info("Creating model images")
    for i, percent in enumerate(percents):
        logging.debug(f"Model image for percent={percent}")
        k = np.argmin(np.abs(percent - total_percent))
        sub_flux = flux[0:(k + 1)]
        sub_Ix = Ix[0:(k + 1)]
        sub_Iy = Iy[0:(k + 1)]
        models[i, sub_Ix, sub_Iy] = sub_flux
        
        logging.info(f"Saving model image to `{true_paths[percent]}`")
        with open(true_paths[percent], "wb") as file:
            np.savez(file, 
                model=models[i],
                Ix=sub_Ix,
                Iy=sub_Iy,
                cell_rad=cell_rad,
                cell_deg=cell_deg,
                cell_asec=cell_asec
            )
    

def create_visibilities(vis_options, params):
    logging.debug("Invoking function")
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
        random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    logging.info("Fetching visibility information")
    sigma_n = vis_options["sigma-n"]
    tol = vis_options["tol"]
    
    logging.debug("Fetching table information")
    with table(str(paths["data"]["ms"]), ack=False) as tb:
        TIME = tb.getcol("TIME")
        ANT1 = tb.getcol("ANTENNA1")
        ANT2 = tb.getcol("ANTENNA2")
        UVW = tb.getcol("UVW")
        
    # Open spectral subtable and retrieve data
    with table(str(paths["data"]["ms"]) + "::SPECTRAL_WINDOW", ack=False) as tb:
        FREQ = tb.getcol("CHAN_FREQ").flatten()

    logging.info("Load the true-gains data")
    true_path = paths["gains"]["true"]["files"]
    gains = load_data(true_path)
    true_gains = gains["gains"][:, :, None, None, None]
    changeAll = False

    percents = params.get("percents", False)
    new_percents = list(map(int, vis_options["percents"].replace(" ", "").split(",")))
    if not percents or set(percents) != set(new_percents):
        with refreeze(params):
            params["percents"] = new_percents

    with refreeze(params):
        params["sigma-n"] = sigma_n
        params["tol"] = tol

    true_paths = paths["fluxes"]["true"]["files"]

    logging.info("Generating visibilities")
    for percent in percents:
        logging.debug("Load skymodel data")
        skymodel = load_data(true_paths[percent]) 
        model = skymodel["model"]
        cell_rad = skymodel["cell_rad"]
        
        logging.debug("Generate model visibilities")
        model_vis = dirty2ms(
                        uvw=UVW, 
                        freq=FREQ, 
                        dirty=model,
                        pixsize_x=cell_rad, 
                        pixsize_y=cell_rad,
                        epsilon=tol, 
                        nthreads=params["n-cpu"],
                        do_wstacking=True
        )[:, None, None]
        
        # Time bin parameters
        _, tbin_indices, tbin_counts = np.unique(TIME, return_index=True, 
                                                    return_counts=True)
        
        logging.debug("Corrupt visibilities")
        clean_vis = corrupt_vis(tbin_indices, tbin_counts, 
                                    ANT1, ANT2, true_gains, model_vis)
        
        logging.debug("Add noise to visibilities")
        noise = np.random.normal(0.0, sigma_n, size=clean_vis.shape)\
                + 1.0j * np.random.normal(0.0, sigma_n, size=clean_vis.shape)
        
        vis = clean_vis + noise
        
        logging.info(f"Saving {percent}MP visibilities to `{paths['data']['ms']}`")
        with table(str(paths["data"]["ms"]), readonly=False, ack=False) as tb:
            
            for column in ["noise", "model", "clean", "data"]:
                desc = tb.getcoldesc("DATA")
                desc["name"] = f"{column.upper()}_{percent}MP"
                desc['comment'] = desc['comment'].replace(" ", "_") 
                
                dminfo = tb.getdminfo("DATA")
                dminfo["NAME"] = f"{column}_{percent}mp"
                
                try:
                    tb.addcols(desc, dminfo)
                except:
                    if changeAll:
                        continue

                    logging.warning(f"Column `{desc['name']}` exists")
                    choice = input(f"Replace `{desc['name']}`? (y/n/a) ")
                    
                    if len(choice) > 0 and choice.lower()[0] == "y":
                        logging.debug("Replace the column")
                        continue
                    elif len(choice) > 0 and choice.lower()[0] == "a":
                        logging.info("Replace all columns")
                        changeAll = True
                        continue
                    else:
                        logging.info("Stopping visibility generation")
                        return
            
            # Add to columns
            logging.debug("Place data in table")
            tb.putcol(f"NOISE_{percent}MP", noise)
            tb.putcol(f"MODEL_{percent}MP", model_vis[..., 0])
            tb.putcol(f"CLEAN_{percent}MP", clean_vis)
            tb.putcol(f"DATA_{percent}MP", vis)

    # Weight Value for all visibilities
    W = 1.0/(2.0 * sigma_n**2)

    with table(str(paths["data"]["ms"]), readonly=False, ack=False) as tb:
        # Create sigma_n column
        desc = tb.getcoldesc("WEIGHT")
        desc["name"] = f"SIGMA_N"
        desc['comment'] = desc['comment'].replace(" ", "_") 
                
        dminfo = tb.getdminfo("WEIGHT")
        dminfo["NAME"] = f"sigma_n"

        try:
            tb.addcols(desc, dminfo)
        except:
            if not changeAll:
                logging.warning(f"Column `{desc['name']}` exists")
                choice = input(f"Replace `{desc['name']}`? (y/n/a) ")
                
                if len(choice) > 0 and choice.lower()[0] == "y":
                    logging.debug("Replace the column")
                elif len(choice) > 0 and choice.lower()[0] == "a":
                    logging.debug("Replace all")
                    changeAll = True
                else:
                    logging.info("Stopping visibility generation")
                    return
            
        weights = tb.getcol("WEIGHT")
        if not np.isclose(weights[0, 0], 0) and not np.isclose(weights[0, 0], 1) and not changeAll:
            logging.warning(f"Column `WEIGHT` has non-zero data")
            choice = input(f"Replace `WEIGHT`? (y/n) ")

            if len(choice) > 0 and choice.lower()[0] == "y":
                pass
            else:
                logging.info("Stopping visibility generation")
                return
                
        logging.debug("Calculating `SIGMA_N` column")
        sigma_array = np.ones_like(weights) * np.sqrt(2) * sigma_n 
        
        logging.debug("Calculating `WEIGHTS` column")
        weight_array = np.ones_like(weights) * W
        
        logging.info("Saving weights to table")
        tb.putcol("SIGMA_N", sigma_array)
        tb.putcol("WEIGHT", weight_array)