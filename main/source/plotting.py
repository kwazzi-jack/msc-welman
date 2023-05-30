import numpy as np
from source.other import check_for_data, cm2in
from source.parameters import Settings, refreeze
from source.data import load_data
from pathlib import Path
import logging
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
from IPython.display import display, clear_output


def plotting_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating plotting options config")
    paths = params["paths"]
    name = "plot-config.yml"
    config_dir = paths["config-dir"]
    with refreeze(paths) as file:
        file["plot-config"] = str(config_dir / name)
    logging.debug(f"Measurement set config at: `{paths['plot-config']}`")

    settings = Settings(
        name=name,
        header="Plotting Options",
        description="""Defines the various settings required for plotting
        the different figures, such as figure size, font size, labels and
        so forth. Note, these are only defaults and can be changed inline.""",
        directory=str(config_dir),
        immutable_path=True,
    )
    logging.debug("Plotting Settings object created")

    settings["label-size"] = (
        "Label font size",
        "The size of the font for labels in the figures.",
        11,
    )
    logging.debug("Setting added: `label-size`")

    settings["title-size"] = (
        "Title font size",
        "The size of the font for titles in the figures.",
        12,
    )
    logging.debug("Setting added: `title-size`")


    settings["tick-size"] = (
        "Tick font size",
        "The size of the font for ticks on axes.",
        10,
    )
    logging.debug("Setting added: `tick-size`")

    settings["dpi"] = (
        "DPI",
        "Determines the quality of the saved figure.",
        150,
    )
    logging.debug("Setting added: `dpi`")

    settings["plot-size"] = (
        "Subplot Size (cm)",
        "The size of a single subplot in centimeters.",
        9,
    )
    logging.debug("Setting added: `plot-size`")

    settings["line-width"] = (
        "Line Width",
        "The width of the curve. Depends on y-axis data.",
        2.0,
    )
    logging.debug("Setting added: `line-width`")

    settings["line-style"] = (
        "Line Style",
        "The style of the curve.",
        "-",
    )
    logging.debug("Setting added: `line-style`")

    app = settings.to_widget()
    if not no_gui:
        clear_output()
        logging.debug("Displaying the settings to notebook")
        display(app)
    try:
        settings.from_yaml(str(Path(name)))
        logging.debug("Plot settings loaded from file")
    except:
        settings.to_yaml(str(Path(name)))
        logging.debug("Plot settings set to default and saved")

    logging.debug("Plot settings complete and returning")
    return settings


def setup_plotting(pl_options, params):
    logging.debug("Invoking function")

    paths = params["paths"]
    logging.info("Retrieving plotting options")
    plot_options = dict()

    label_size = pl_options["label-size"]
    plot_options["label-size"] = label_size
    logging.debug(f"Identified option: `label-size={label_size}`")
    
    title_size = pl_options["title-size"]
    plot_options["title-size"] = title_size
    logging.debug(f"Identified option: `title-size={title_size}`")

    tick_size = pl_options["tick-size"]
    plot_options["tick-size"] = tick_size
    logging.debug(f"Identified option: `tick-size={tick_size}`")

    dpi = pl_options["dpi"]
    plot_options["dpi"] = dpi
    logging.debug(f"Identified option: `dpi={dpi}`")

    plot_size = pl_options["plot-size"]
    plot_options["size"] = plot_size
    logging.debug(f"Identified option: `plot-size={plot_size}`")

    line_width = pl_options["line-width"]
    plot_options["line-width"] = line_width
    logging.debug(f"Identified option: `line-width={line_width}`")

    line_style = pl_options["line-style"]
    plot_options["line-style"] = line_style
    logging.debug(f"Identified option: `line-style={line_style}`")

    plot_options["kalcal"] = {
        "name" : r"\texttt{kalcal}",
        "scale" : "log",
        "label" : r"$\log_{10}(\sigma_f)$",
        "energy" : {
            "label" : r"$\varphi_k$"
        },
        "diagonal": {
            "name" : r"\texttt{kalcal-diag}",
            "filter" : {
                "name" : r"\texttt{diag-filter}",
                "color" : "#ff7f0e",
                "line-width" : line_width,
                "line-style" : ":"
            },
            "smoother" : {
                "name" : r"\texttt{diag-smoother}",
                "color" : "#1f77b4",
                "line-width" : line_width,
                "line-style" : line_style
            },
            "energy" : {
                "name" : r"\texttt{diag-}$\varphi_k$",
                "color" : "indigo",
                "line-width" : line_width,
                "line-style" : "--"
            }
        },
        "full": {
            "name" : r"\texttt{kalcal-full}",
            "filter" : {
                "name" : r"\texttt{full-filter}",
                "color" : "#ff7f0e",
                "line-width" : line_width,
                "line-style" : ":"
            },
            "smoother" : {
                "name" : r"\texttt{full-smoother}",
                "color" : "#2ca02c",
                "line-width" : line_width,
                "line-style" : line_style
            },
            "energy" : {
                "name" : r"\texttt{full-}$\varphi_k$",
                "color" : "indigo",
                "line-width" : line_width,
                "line-style" : "--"
            }
        }
    }
    logging.debug("Adding `kalcal` plot options")

    plot_options["quartical"] = {
        "name" : r"\texttt{QuartiCal}",
        "label" : r"$\Delta_t$",
        "scale" : "linear",
        "color" : "#d62728",
        "line-width" : line_width,
        "line-style" : "-"
    }
    logging.debug("Adding `quartical` plot options")
    
    import scienceplots
    plt.style.use(["science", "no-latex", "nature"])
    logging.debug("`SciencePlots` module loaded, no-latex, nature mode")

    logging.info("Updating parameter data")
    with refreeze(params):
        params["plots"] = plot_options
        logging.debug("Parameter added: `plots`")

def place_ticks(axes, tick_type=None):
    for ax in axes:
        if isinstance(tick_type, (int, float)):
            ax.xaxis.set_major_locator(MultipleLocator(tick_type))
        elif "kalcal" in tick_type:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter("{x:.0f}") 
        elif "quartical" in tick_type:
            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_major_formatter("{x:.0f}")
        else:
            ax.xaxis.set_major_locator(AutoLocator())
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

def init_plot(params, title, algorithms, layout=(2, 2), size=None):
    logging.debug("Invoking function")
    logging.debug("Determine plot options")
    plot_options = params["plots"]

    if size:
        plot_size = size
        logging.debug("Custom subplot size")
    else:
        plot_size = plot_options["size"]
        logging.debug("Default subplot size")
        
    n_rows, n_cols = layout
    label_size = plot_options["label-size"]
    title_size = plot_options["title-size"]
    tick_size = plot_options["tick-size"]

    logging.info("Setting up axes and figure")
    logging.debug(f"Subplots shape: (N_row, N_cols) = ({n_rows}, {n_cols})")
    figsize = (cm2in(plot_size * n_rows), cm2in(plot_size * n_cols))
    logging.debug(f"Subplots size: {figsize[0]}x{figsize[1]}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    aux = []
    logging.debug("Axes and figures created")

    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    fig.suptitle(title, size=title_size, x=mid)
    logging.debug("Setting figure title")

    logging.info("Configuring axes")
    for i, ax in enumerate(axes):
        ax.set_prop_cycle(None)
        ax.tick_params(axis='both', which="both", labelsize=tick_size)

        if "kalcal" in algorithms:
            logging.debug(f"Setup `kalcal` axes={i}")
            ax.set_xlabel(plot_options["kalcal"]["label"], size=label_size)
            ax.set_xscale(plot_options["kalcal"]["scale"])
        if "quartical" in algorithms:
            logging.debug(f"Setup `quartical` twin y-axis={i}")
            ax2 = ax.twiny()
            ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
            ax2.set_xlabel(plot_options["quartical"]["label"], size=label_size)
            ax2.set_xscale(plot_options["quartical"]["scale"])
            ax2.invert_xaxis()
            ax2.tick_params(axis='x', which="both", labelsize=tick_size)
            aux.append(ax2)
        if "energy" in algorithms:
            logging.debug(f"Setup `energy` twin x-axis={i}")
            ax2 = ax.twinx()
            ax2.set_xscale(plot_options["kalcal"]["scale"])
            ax2.set_ylabel(plot_options["energy"]["label"], size=label_size)
            ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
            ax2.tick_params(axis='y', which="both", labelsize=tick_size)
            aux.append(ax2) 
        if "time" in algorithms:
            logging.debug(f"Setup `time` axes={i}")
            ax.set_xlabel(r"Time Step, $k$", size=label_size)
            ax.set_xscale("linear")
        if "source" in algorithms:
            logging.debug(f"Setup `source` axes={i}")
            ax.set_xlabel("Source Index", size=label_size)
            ax.set_xscale("linear")

    if ("kalcal" in algorithms and "quartical" in algorithms) \
    or ("kalcal" in algorithms and "energy" in algorithms):
        logging.debug(f"Adjust subplots for twin axes")       
        top = 0.85 if "\n" in title else 0.875
        hspace = 0.35
        wspace = 0.35
        right = 0.9
    else:
        logging.debug(f"Adjust subplots for single axes")
        top = 0.875 if "\n" in title else 0.9
        hspace = 0.55
        wspace = 0.30
        right = None

    if "time" in algorithms or "source" in algorithms:
        logging.debug("No legend adjust")
        bottom = None
    else:
        bottom = 0.125

    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, 
                        wspace=wspace, right=right)

    logging.info(f"Figure and axes setup complete")
    return fig, axes, aux


def create_amplitude_and_phase_signal_plot(ants, params, show=False):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    plot_path = paths["plots-dir"] / "ch5-sim-gains.png"

    try:
        logging.debug("Checking for existing plot")
        check_for_data(plot_path)
    except:
        logging.info("No plotting done")
        return
    
    logging.info("Load the true-gains data")
    true_path = paths["gains"]["true"]
    gains = load_data(true_path)
    
    amp_gains = gains["amp_gains"]
    phase_gains = gains["phase_gains"]        
    n_rows, n_cols = len(ants), 2

    logging.info("Creating plot")
    fig, axes, _ = init_plot(
        params,
        title="Amplitude and Phase Components of True Gains Signal",
        algorithms="time",
        layout=(n_rows, n_cols),
        size=9
    )
    
    label_size = plot_options["label-size"]
    line_width = plot_options["line-width"]
    line_style = plot_options["line-style"]
    times = np.arange(1, params["n-time"] + 1)

    logging.info("Plotting amplitude and phase signals")
    for i, ant in enumerate(ants):       
        c = 2 * i

        logging.debug(f"Plot amplitude over time for `ant={ant}`")
        apq = amp_gains[:, ant]
        axes[c].plot(times, apq.real, color="red", 
                     ls=line_style, lw=line_width)
        axes[c].set_ylabel(rf"$A_{ant}(t)$", size=label_size)
        
        logging.debug(f"Plot phase over time for `ant={ant}`")
        ppq = phase_gains[:, ant]
        axes[c + 1].plot(times, ppq.real, color="green", 
                     ls=line_style, lw=line_width)
        axes[c + 1].set_ylabel(rf"$\phi_{ant}(t)$", size=label_size)

    # Place ticks    
    place_ticks(axes, params["n-time"]//6)
    logging.debug("Placing ticks")

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])

    if show:
        logging.info("Displaying amplitude and phase signal plot")
        plt.show()