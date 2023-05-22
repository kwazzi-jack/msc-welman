from pathlib import Path
import os
from source.parameters import Settings, refreeze, YamlDict
from source.other import check_for_data
from IPython.display import display, clear_output
import logging
import time


def script_settings(path="config.yml"):
    directory, name = os.path.split(path)
    settings = Settings(
        name=str(name),
        header="Script Settings",
        description="""Relates to all the system settings relating to the script.
        This covers the entire script from top to bottom and
        so please set carefully. Please correctly set these parameters
        to ensure no over usage of processing, no incorrect paths and
        replicability of data terms.""",
        directory=Path(directory),
        immutable_path=False,
    )

    settings["name"] = (
        "Identifier/Name",
        "Specific name to assign to a simulation run.",
        "meerkat-src-100",
    )

    settings["config-dir"] = (
        "Config file directory",
        "Folder for all generated configuration files. If empty, then `name/config`.",
        "",
    )

    settings["data-dir"] = (
        "Data directory",
        "Path to directory to hold data files. If empty, then `name/data`.",
        "",
    )

    settings["plots-dir"] = (
        "Plots directory",
        "Directory to place figures, images and plots. If empty, then `name/plots`.",
        "",
    )

    settings["mpl-dir"] = (
        "matplotlib config directory",
        "Path to matplotlib config folder. If empty, then auto-select.",
        "",
    )

    settings["n-cpu"] = (
        "Number of CPU Cores",
        "How many cores the script is allowed to use for calculations. Default: 8.",
        8,
    )

    settings["seed"] = (
        "Random state seed",
        "Pre-defined seed to reproduce random values. Default: 666.",
        666,
    )

    settings["log-level"] = (
        "Logging Level",
        "Determines how much information is logged to file. Default: 1-ERROR.",
        ["1-ERROR", "2-WARNING", "3-INFO", "4-DEBUG"],
    )

    clear_output()
    display(settings.to_widget())
    try:
        settings.from_yaml(name)
    except:
        settings.to_yaml(name)

    return settings


def __setup_logging(options):
    name = options["name"]
    log_path = Path(f"{name}.log")
    log_path.unlink(missing_ok=True)
    level = options["log-level"]

    logging.root.handlers.clear()
    if "error" in level.lower():
        level = logging.ERROR
    elif "warning" in level.lower():
        level = logging.WARNING
    elif "info" in level.lower():
        level = logging.INFO
    elif "debug" in level.lower():
        level = logging.DEBUG

    logging.basicConfig(level=logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s | %(levelname)s | %(module)s.%(funcName)s] %(message)s"
    )

    logger = logging.getLogger()
    stream_handler = logger.handlers[0]
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def __generate_dir_skeleton(paths):
    logging.info("Creating directory paths")
    main_dir = paths["main-dir"]
    config_dir = paths["config-dir"]
    data_dir = paths["data-dir"]
    plots_dir = paths["plots-dir"]
    runs_dir = paths["runs-dir"]

    dirs = [main_dir, config_dir, data_dir, plots_dir, runs_dir]

    for mtype in ["gains", "fluxes", "fits"]:
        base_dir = data_dir / mtype
        dirs.append(base_dir)
        logging.debug(f"Added to paths: `{base_dir}`")
        paths[mtype] = dict()
        for alg in ["kalcal-diag", "kalcal-full", "quartical", "true"]:
            alg_dir = base_dir / alg
            dirs.append(alg_dir)
            logging.debug(f"Added to paths: `{alg_dir}`")
            if "kalcal" in alg:
                filter_path = alg_dir / "filter"
                smoother_path = alg_dir / "smoother"
                paths[mtype][alg] = {"filter": filter_path, "smoother": smoother_path}
                dirs.append(filter_path)
                logging.debug(f"Added to paths: `{filter_path}`")
                dirs.append(smoother_path)
                logging.debug(f"Added to paths: `{smoother_path}`")
            else:
                paths[mtype][alg] = alg_dir

    logging.info("Making directories")
    for path in dirs:
        try:
            os.makedirs(path)
            while not os.path.exists(path):
                time.sleep(0.1)
            logging.debug(f"Created: `{path}`")
        except OSError:
            logging.debug(f"Exists, do nothing: `{path}`")

    refreeze(paths)
    logging.info("Path data saved to file")


def __set_thread_count(options):
    n_cpu = "8" if options["n-cpu"] == None else str(options["n-cpu"])
    logging.info(f"Setting thread count to {n_cpu}")

    for env_var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
        try:
            logging.debug(f"Original `{env_var}`={os.environ[env_var]}")
        except KeyError:
            logging.debug(f"No `{env_var}` environment variable found")

        os.environ[env_var] = n_cpu
        logging.debug(f"Setting `{env_var}` to {n_cpu}")


def __set_random_seed(options):
    import numpy, random

    seed = 666 if options["seed"] == None else options["seed"]
    logging.info(f"Setting random state seed to {seed}")
    numpy.random.seed(seed)
    logging.debug(f"`numpy` seed set to {seed}")
    random.seed(seed)
    logging.debug(f"`random` seed set to {seed}")


def __set_matplotlib_dir(options):
    dirname = options["mpl-dir"]
    if len(dirname):
        logging.info(f"Setting matplotlib config directory to {dirname}")
        os.environ["MPLCONFIGDIR"] = dirname
    else:
        logging.info(f"Matplotlib config directory unchanged")


def __set_paths(options):
    logging.info("Fetching directory paths")
    main_dir = Path(options["name"])
    logging.debug(f"Identified main directory: `{main_dir}`")
    config_dir = (
        main_dir / "config"
        if len(options["config-dir"]) == 0
        else Path(options["config-dir"])
    )
    logging.debug(f"Identified config directory: `{config_dir}`")
    data_dir = (
        main_dir / "data"
        if len(options["data-dir"]) == 0
        else Path(options["data-dir"])
    )
    logging.debug(f"Identified data directory: `{data_dir}`")
    plots_dir = (
        main_dir / "plots"
        if len(options["plots-dir"]) == 0
        else Path(options["plots-dir"])
    )
    logging.debug(f"Identified plots directory: `{plots_dir}`")
    runs_dir = data_dir / "runs"
    logging.debug(f"Identified quartical runs directory: `{runs_dir}`")
    path_name = config_dir / "paths.yml"
    logging.debug(f"Identified path file: `{path_name}`")
    log_name = options["name"] + ".log"
    logging.debug(f"Identified log file: `{log_name}`")

    logging.info("Creating new path data")
    path_config = YamlDict(path_name, freeze=True, overwrite=True)
    logging.debug("Path config file created, set to frozen")

    logging.debug("Setting path data as previously identified")
    path_config["main-dir"] = main_dir
    path_config["config-dir"] = config_dir
    path_config["data-dir"] = data_dir
    path_config["plots-dir"] = plots_dir
    path_config["runs-dir"] = runs_dir
    path_config["log"] = log_name
    path_config["main-config"] = options.path

    logging.info("Completed and returning paths")
    return path_config


def __set_params(paths, options):
    config_dir = paths["config-dir"]
    path_name = config_dir / "params.yml"
    logging.info("Creating new parameter data")
    param_config = YamlDict(path_name, freeze=True, overwrite=True)
    logging.debug("Parameter config file created, set to frozen")
    logging.debug("Setting parameter data")

    with refreeze(param_config) as file:
        file["n-cpu"] = options["n-cpu"]
        logging.debug(f"Parameter added: `n-cpu={options['n-cpu']}`")
        file["seed"] = options["seed"]
        logging.debug(f"Parameter added: `seed={options['seed']}`")
        file["paths"] = paths
        logging.debug(f"Path data moved into parameter config")

    logging.info("Completed and returning parameters")
    return param_config

def setup_simulation(options):
    __setup_logging(options)
    paths = __set_paths(options)
    __generate_dir_skeleton(paths)
    __set_thread_count(options)
    __set_random_seed(options)
    __set_matplotlib_dir(options)
    params = __set_params(paths, options)
    logging.info("Simulation setup complete")
    return params
