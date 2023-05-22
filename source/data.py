from source.parameters import Settings, YamlDict, refreeze
from pathlib import Path
from ipywidgets import Output
from IPython.display import display, clear_output
from subprocess import Popen
from source.other import check_for_data, DataExistsError
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
from africanus.coordinates import radec_to_lm
from casacore.tables import table
import numpy as np
import logging


def measurement_set_options(params):
    logging.debug("Creating measurement set options config")
    paths = params["paths"]
    name = "ms-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["ms-config"] = str(Path(config_dir) / name)
    logging.debug(f"Measurement set config at: `{paths['ms-config']}`")

    settings = Settings(
        name=name,
        header="Measurement Set",
        description="""Settings to generate an empty measurement set for the simulation.
        It leverages the software `simms` to create a suitable database for this experiment.""",
        directory=str(config_dir),
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

    clear_output()
    logging.debug("Displaying the settings to notebook")
    display(settings.to_widget())
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("Measurement set settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("Measurement set settings set to default and saved")

    logging.debug("Measurement set settings complete and returning")
    return settings


def create_empty_measurement_set(ms_options, params, verbose=False):
    paths = params["paths"]

    logging.info("Retrieving measurement set options")

    telescope = str(ms_options["telescope"]).lower()
    logging.debug(f"Identified option: `telescope={telescope}`")

    ms_name = ms_options["ms-name"] if len(ms_options["ms-name"]) else f"{telescope}.ms"
    logging.debug(f"Identified option: `ms-name={ms_name}`")

    ra, dec = ms_options["ra", "dec"]
    logging.debug(f"Identified option: `ra={ra}`")
    logging.debug(f"Identified option: `dec={dec}`")

    st, dt = ms_options["synth-time", "d-time"]
    logging.debug(f"Identified option: `synth-time={st}`")
    logging.debug(f"Identified option: `d-time={dt}`")

    freq_0 = ms_options["freq-0"]
    logging.debug(f"Identified option: `freq-0={freq_0}`")

    data_dir = paths["data-dir"]
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
        with refreeze(paths) as file:
            file["ms-path"] = ms_path
            logging.debug(f"Added to paths: `{ms_path}`")

        logging.debug("Opening measurement set")
        with table(str(ms_path), ack=False) as tb:
            ANT1 = tb.getcol("ANTENNA1")
            ANT2 = tb.getcol("ANTENNA2")
            logging.debug("Antenna columns pulled")

        n_ant = np.max((ANT1.max(), ANT2.max())) + 1
        logging.debug(f"Calculated number of antennas as {n_ant}")

        n_time = st * 3600 // dt
        logging.debug(f"Calculated number of time-steps as {n_time}")

        with refreeze(params) as file:
            file["n-ant"] = n_ant
            logging.debug(f"Parameter added: `n-ant={n_ant}`")
            file["n-time"] = n_time
            logging.debug(f"Parameter added: `n-time={n_time}`")
            file["n-chan"] = 1
            logging.debug(f"Parameter added: `n-chan=1`")
            file["n-dir"] = 1
            logging.debug(f"Parameter added: `n-dir=1`")
            file["n-corr"] = 1
            logging.debug(f"Parameter added: `n-corr=1`")

        logging.info(f"New measurement set created at `{ms_name}`")
    except DataExistsError:
        logging.info(f"Keep original measurement set at `{ms_name}`")


def gains_options(params):
    logging.debug("Creating gains options config")
    paths = params["paths"]

    name = "gains-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["gains-config"] = config_dir / name    
    logging.debug(f"Gains options config at: `{paths['ms-config']}`")
    
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
        directory=config_dir,
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

    clear_output()
    logging.debug("Displaying the settings to notebook")
    display(settings.to_widget())
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


def create_gains_signal(gains_options, params):
    paths = params["paths"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")
    
    if str(paths["gains"]["true"])[-4:] == ".npy":
        true_path = paths["gains"]["true"]
        logging.debug("True gains path exists, do nothing")
    else:
        true_path = paths["gains"]["true"] / "true-gains.npy"
        logging.debug(f"Added to paths: `{true_path}`")

    try:
        logging.debug("Checking for existing true gains")
        check_for_data(true_path)
    except:
        logging.info("No deletion done")
        return

    # Open main table and retrieve data
    logging.info(f"Retrieving information from `{paths['ms-path']}`")
    with table(str(paths["ms-path"]), ack=False) as tb:
        TIME = tb.getcol("TIME")
        ANT1 = tb.getcol("ANTENNA1")
        ANT2 = tb.getcol("ANTENNA2")
        logging.debug("Retrieved antenna phase directinand time columns")

    # Open field subtable and retrieve data
    with table(str(paths["ms-path"]) + "::FIELD", ack=False) as tb:
        PHASE_DIR = tb.getcol("PHASE_DIR").astype(np.float64)
        logging.debug("Retrieved phase directin column")
        
    # Open spectral subtable and retrieve data
    with table(str(paths["ms-path"]) + "::SPECTRAL_WINDOW", ack=False) as tb:
        FREQ = tb.getcol("CHAN_FREQ").flatten()[0].astype(np.float64)
        logging.debug("Retrieved frequency column")
        
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

    # Create lm-array
    lm = np.array(radec_to_lm(PHASE_DIR.reshape(1, -1)))

    # Axes ranges
    axes = (tbin_indices, FREQ, lm)

    # Amplitude Parameters
    amp_scales = (gains_options["amp-length"], 1.0, 1.0)
    amp_stds = (gains_options["amp-var"], 1.0, 1.0)
    logging.debug("Amplitude information set")

    # Phase Parameters
    phase_scales = (gains_options["phase-length"], 1.0, 1.0)
    phase_stds = (gains_options["phase-var"], 1.0, 1.0)
    logging.debug("Phase information set")
    
    # Make complex-gains using above
    if gains_options["type"].lower() == "full-complex":
        logging.info("Generating full-complex gains")
        amp_gains = __create_gains(
            axes, amp_scales, amp_stds, shape, mean=gains_options["amp-mean"]
        )
        logging.debug("Amplitude signal generated")
        phase_gains = __create_gains(
            axes, phase_scales, phase_stds, shape, mean=gains_options["phase-mean"]
        )
        logging.debug("Phase signal generated")
        complex_gains = amp_gains * np.exp(1.0j * phase_gains)
        logging.debug("Final complex gains created from amplitude and phase")
    elif gains_options["type"].lower() == "amplitude-only":
        logging.info("Generating amplitude-only gains")
        amp_gains = __create_gains(
            axes, amp_scales, amp_stds, shape, mean=gains_options["amp-mean"]
        )
        logging.debug("Amplitude signal generated")
        phase_gains = np.ones(shape)
        logging.debug("Phase is unity")
        complex_gains = amp_gains * np.exp(1.0j * phase_gains)
        logging.debug("Final complex gains created from amplitude and phase")
    elif gains_options["type"].lower() == "phase-only":
        logging.info("Generating phase-only gains")
        amp_gains = np.ones(shape)
        logging.debug("Amplitude is unity")
        phase_gains = __create_gains(
            axes, phase_scales, phase_stds, shape, mean=gains_options["phase-mean"]
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
        np.save(file, complex_gains)
        np.save(file, amp_gains)
        np.save(file, phase_gains)

    logging.info("Updating path data")
    with refreeze(paths) as file:
        paths["gains"]["true"] = true_path

    logging.info(f"New simulated gains at `{paths['gains']['true']}`")


def load_gains(path):
    logging.info(f"Loading gains from `{path}`")
    item = np.load(path)
    if item.ndim == 5:
        logging.debug("Codex form shape, reducing to antenna and time")
        return item[:, :, 0, 0, 0]
    elif item.ndim == 6:
        logging.debug("Kalcal form shape, reducing to antenna and time")
        return item[:, :, 0, 0, 0, 0]
    else:
        logging.error("Unknown gains shape used")
        raise ValueError("Unknown gains data format used.")