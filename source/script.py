from pathlib import Path
import os
from source.parameters import Settings, refreeze, YamlDict
from source.other import check_for_data, DataExistsError
from source.plotting import plotting_options
from source.data import measurement_set_options, gains_options, visibility_options
from source.algorithms import kalcal_diag_options, kalcal_full_options, quartical_options
from IPython.display import display, clear_output
import logging
import time
import shutil
import ipywidgets as widgets

def script_settings(path="config.yml", no_gui=False):
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

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        display(app)
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
        "[%(asctime)s | %(levelname)s | source.%(module)s] %(message)s"
    )

    logger = logging.getLogger()
    stream_handler = logger.handlers[0]
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    formatter = logging.Formatter(
        "[%(asctime)s | %(levelname)s | %(module)s.%(funcName)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def __generate_dir_skeleton(params, cleanup=False):
    logging.info("Creating directory paths")
    paths = params["paths"]

    dirs = [
        paths["main"]["dir"], 
        paths["config"]["dir"], 
        paths["data"]["dir"],
        paths["gains"]["dir"],
        paths["gains"]["true"]["dir"],
        paths["gains"]["kalcal-diag"]["filter"]["dir"],
        paths["gains"]["kalcal-diag"]["smoother"]["dir"],
        paths["gains"]["kalcal-full"]["filter"]["dir"],
        paths["gains"]["kalcal-full"]["smoother"]["dir"],
        paths["gains"]["quartical"]["dir"],
        paths["fluxes"]["dir"],
        paths["fluxes"]["true"]["dir"],
        paths["fluxes"]["kalcal-diag"]["filter"]["dir"],
        paths["fluxes"]["kalcal-diag"]["smoother"]["dir"],
        paths["fluxes"]["kalcal-full"]["filter"]["dir"],
        paths["fluxes"]["kalcal-full"]["smoother"]["dir"],
        paths["fluxes"]["quartical"]["dir"],
        paths["fits"]["dir"],
        paths["fits"]["true"]["dir"],
        paths["fits"]["kalcal-diag"]["filter"]["dir"],
        paths["fits"]["kalcal-diag"]["smoother"]["dir"],
        paths["fits"]["kalcal-full"]["filter"]["dir"],
        paths["fits"]["kalcal-full"]["smoother"]["dir"],
        paths["fits"]["quartical"]["dir"], 
        paths["plots"]["dir"], 
        paths["runs"]["dir"]
    ]

    if cleanup:
        try:
            check_for_data(*dirs)
            for path in dirs:
                try:
                    shutil.rmtree(path)
                except:
                    pass
        except DataExistsError:
            pass

    logging.info("Making directories")
    for path in dirs:
        try:
            os.makedirs(path)
            while not path.exists():
                time.sleep(0.1)
            logging.debug(f"Created: `{path}`")
        except OSError:
            logging.debug(f"Exists, do nothing: `{path}`")

def __set_thread_count(params):
    n_cpu = "8" if params["n-cpu"] == None else str(params["n-cpu"])
    logging.info(f"Setting thread count to {n_cpu}")

    for env_var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
        try:
            logging.debug(f"Original `{env_var}`={os.environ[env_var]}")
        except KeyError:
            logging.debug(f"No `{env_var}` environment variable found")

        os.environ[env_var] = n_cpu
        logging.debug(f"Setting `{env_var}` to {n_cpu}")


def __set_random_seed(params):
    import numpy, random

    seed = 666 if params["seed"] == None else params["seed"]
    logging.info(f"Setting random state seed to {seed}")
    numpy.random.seed(seed)
    logging.debug(f"`numpy` seed set to {seed}")
    random.seed(seed)
    logging.debug(f"`random` seed set to {seed}")


def __set_matplotlib_dir(params):
    dirname = params["mpl-dir"]
    if len(dirname):
        logging.info(f"Setting matplotlib config directory to {dirname}")
        os.environ["MPLCONFIGDIR"] = dirname
    else:
        logging.info(f"Matplotlib config directory unchanged")

def __set_params(options):
    logging.info("Fetching directory paths")
    main_dir = Path(options["name"])
    config_dir = (
        main_dir / "config"
        if len(options["config-dir"]) == 0
        else Path(options["config-dir"])
    )
    data_dir = (
        main_dir / "data"
        if len(options["data-dir"]) == 0
        else Path(options["data-dir"])
    )
    plots_dir = (
        main_dir / "plots"
        if len(options["plots-dir"]) == 0
        else Path(options["plots-dir"])
    )
    runs_dir = data_dir / "runs"
    path_name = config_dir / "params.yml"

    try:
        check_for_data(path_name)
        logging.info("Replacing parameter data")
    except DataExistsError:
        logging.info("Using saved parameter data")
        return YamlDict(path_name, freeze=True)
    
    logging.info("Creating new parameter data")
    params = YamlDict(path_name, freeze=True, overwrite=True)
    logging.debug("Parameter config file created, set to frozen")
    
    logging.info("Setting path information")

    params["paths"] = {
        "main": {"dir": main_dir, "files": {}},
        "config": {"dir": config_dir, "files": {}},
        "options": {"dir": config_dir, "files": {}},
        "data": {"dir": data_dir, "files": {}},
        "gains": {
            "dir": data_dir / "gains",
            "true": {
                "dir": data_dir / "gains" / "true",
                "template" : "true-gains.npz",
                "files" : {}
            },
            "kalcal-diag": {
                "dir": data_dir / "gains" / "kalcal-diag",
                "filter": {
                    "dir": data_dir / "gains" / "kalcal-diag" / "filter",
                    "template" : "diag-filter-gains-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files" : {}
                },
                "smoother": {
                    "dir": data_dir / "gains" / "kalcal-diag" / "smoother",
                    "template" : "diag-smoother-gains-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files" : {}
                }
            },
            "kalcal-full": {
                "dir": data_dir / "gains" / "kalcal-full",
                "filter": {
                    "dir": data_dir / "gains" / "kalcal-full" / "filter",
                    "template" : "full-filter-gains-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files" : {}
                },
                "smoother": {
                    "dir": data_dir / "gains" / "kalcal-full" / "smoother",
                    "template" : "full-smoother-gains-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files" : {}
                }
            },
            "quartical": {
                "dir": data_dir / "gains" / "quartical",
                "template" : "quartical-gains-{mp}mp-t_int-{t_int}.npz",
                "files" : {}
            }
        },
        "fluxes": {
            "dir": data_dir / "fluxes",
            "true": {
                "dir": data_dir / "fluxes" / "true",
                "template": "model-{mp}mp.npz",
                "files": {}
            },
            "kalcal-diag": {
                "dir": data_dir / "fluxes" / "kalcal-diag",
                "filter": {
                    "dir": data_dir / "fluxes" / "kalcal-diag" / "filter",
                    "template" : "diag-filter-fluxes-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files": {}
                },
                "smoother": {
                    "dir": data_dir / "fluxes" / "kalcal-diag" / "smoother",
                    "template" : "diag-smoother-fluxes-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files": {}
                    }
            },
            "kalcal-full": {
                "dir": data_dir / "fluxes" / "kalcal-full",
                "filter": {
                    "dir": data_dir / "fluxes" / "kalcal-full" / "filter",
                    "template" : "full-filter-fluxes-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files": {}
                    },
                "smoother": {
                    "dir": data_dir / "fluxes" / "kalcal-full" / "smoother",
                    "template" : "full-smoother-fluxes-{mp}mp-sigma_f-{sigma_f}.npz",
                    "files": {}
                    }
            },
            "quartical": {
                "dir": data_dir / "fluxes" / "quartical",
                "template" : "quartical-fluxes-{mp}mp-t_int-{t_int}.npz",
                    "files": {}
            }
        },
        "fits": {
            "dir": data_dir / "fits",
            "true": {
                "dir": data_dir / "fits" / "true",
                "template": "true-{itype}-{mp}mp.fits",
                "files": {}
            },
            "kalcal-diag": {
                "dir": data_dir / "fits" / "kalcal-diag",
                "filter": {
                    "dir": data_dir / "fits" / "kalcal-diag" / "filter",
                    "template" : "diag-filter-{itype}-{mp}mp-sigma_f-{sigma_f}.fits",
                    "files": {}
                },
                "smoother": {
                    "dir": data_dir / "fits" / "kalcal-diag" / "smoother",
                    "template" : "diag-smoother-{itype}-{mp}mp-sigma_f-{sigma_f}.fits",
                    "files": {}
                    }
            },
            "kalcal-full": {
                "dir": data_dir / "fits" / "kalcal-full",
                "filter": {
                    "dir": data_dir / "fits" / "kalcal-full" / "filter",
                    "template" : "full-filter-{itype}-{mp}mp-sigma_f-{sigma_f}.fits",
                    "files": {}
                    },
                "smoother": {
                    "dir": data_dir / "fits" / "kalcal-full" / "smoother",
                    "template" : "full-smoother-{itype}-{mp}mp-sigma_f-{sigma_f}.fits",
                    "files": {}
                    }
            },
            "quartical": {
                "dir": data_dir / "fits" / "quartical",
                "template" : "quartical-{itype}-{mp}mp-t_int-{t_int}.fits",
                    "files": {}
            }
        },
        "runs": {
            "dir": runs_dir, 
            "template": "quartical-run-{mp}mp-t_int-{t_int}", 
            "files": {}
            },
        "plots": {"dir": plots_dir, "files": {}}
    }

    logging.info("Fetching script options")
    params["mpl-dir"] = options["mpl-dir"]
    params["seed"] = options["seed"]
    params["n-cpu"] = options["n-cpu"]

    refreeze(params)
    logging.info("Completed and returning parameters")
    return params

def setup_simulation(options, cleanup=False):
    __setup_logging(options)
    params = __set_params(options)
    __generate_dir_skeleton(params, cleanup)
    __set_thread_count(params)
    __set_random_seed(params)
    __set_matplotlib_dir(params)
    logging.info("Simulation setup complete")
    return params

def main_settings(name="config.yml", cleanup=False):
    titles = ['Script', 'Plotting', 'Measurement Set', 
              'Gains', 'Visibilities', "kalcal-diag",
              "kalcal-full", "QuartiCal"]
    script_options = script_settings(name, no_gui=True)
    params = setup_simulation(script_options, cleanup)

    pl_options = plotting_options(params, no_gui=True)
    ms_options = measurement_set_options(params, no_gui=True)
    gs_options = gains_options(params, no_gui=True)
    vis_options = visibility_options(params, no_gui=True)
    kld_options = kalcal_diag_options(params, no_gui=True)
    klf_options = kalcal_full_options(params, no_gui=True)
    qut_options = quartical_options(params, no_gui=True)

    children = [
        script_options.to_widget(),
        pl_options.to_widget(),
        ms_options.to_widget(),
        gs_options.to_widget(),
        vis_options.to_widget(),
        kld_options.to_widget(),
        klf_options.to_widget(),
        qut_options.to_widget()
    ]

    tab = widgets.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, titles[i])

    options = {
        "script" : script_options,
        "plots" : pl_options,
        "gains" : gs_options,
        "ms" : ms_options,
        "vis" : vis_options,
        "kalcal-diag" : kld_options,
        "kalcal-full" : klf_options,
        "quartical" : qut_options
    }

    display(tab)
    return params, options
