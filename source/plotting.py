from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from numpy.typing import NDArray
from source.other import check_for_data, cm2in
from source.parameters import Settings
from source.data import load_data
from pathlib import Path
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, AutoLocator, AutoMinorLocator
from IPython.display import display, clear_output
import matplotlib.colors as mc
import colorsys
from source.metrics import (
    get_energy_values,
    calculate_mse_on_gains,
    calculate_mse_on_fluxes,
    calculate_total_ratio_on_fluxes,
    calculate_psi_0_on_fluxes,
    calculate_psi_1_on_fluxes,
    calculate_rms_from_residuals,
    calculate_rms_from_corrected_residuals,
    percent_flux_error,
    flux_ratio,
    flux_error
)

def adjust_brightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def plotting_options(params, no_gui=False):
    logging.debug("Invoking function")
    
    logging.debug("Creating plotting options config")
    paths = params["paths"]
    name = "plot-options.yml"
    options_dir = paths["options"]["dir"]
    paths["options"]["plots"] = options_dir / name
    params.save()
    logging.debug(f"Plotting options config at: `{options_dir / name}`")

    settings = Settings(
        name=name,
        header="Plotting Options",
        description="""Defines the various settings required for plotting
        the different figures, such as figure size, font size, labels and
        so forth. Note, these are only defaults and can be changed inline.""",
        directory=options_dir,
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

    logging.info("Retrieving plotting options")
    plot_options = dict()

    label_size = pl_options["label-size"]
    plot_options["label-size"] = label_size
    
    title_size = pl_options["title-size"]
    plot_options["title-size"] = title_size

    tick_size = pl_options["tick-size"]
    plot_options["tick-size"] = tick_size

    dpi = pl_options["dpi"]
    plot_options["dpi"] = dpi

    plot_size = pl_options["plot-size"]
    plot_options["size"] = plot_size

    line_width = pl_options["line-width"]
    plot_options["line-width"] = line_width

    line_style = pl_options["line-style"]
    plot_options["line-style"] = line_style

    plot_options["kalcal"] = {
        "name" : r"kalcal",
        "scale" : "log",
        "label" : r"$\log_{10}(\sigma_f)$",
        "energy" : {
            "label" : r"$\log_{10}(\varphi_k)$"
        },
        "diagonal": {
            "name" : r"kalcal-diag",
            "filter" : {
                "name" : r"diag-filter",
                "color" : "#ff7f0e",
                "line-width" : line_width,
                "line-style" : ":"
            },
            "smoother" : {
                "name" : r"diag-smoother",
                "color" : "#1f77b4",
                "line-width" : line_width,
                "line-style" : line_style
            },
            "energy" : {
                "name" : r"diag-$\varphi_k$",
                "color" : "indigo",
                "line-width" : line_width,
                "line-style" : "--"
            }
        },
        "full": {
            "name" : r"kalcal-full",
            "filter" : {
                "name" : r"full-filter",
                "color" : "#ff7f0e",
                "line-width" : line_width,
                "line-style" : ":"
            },
            "smoother" : {
                "name" : r"full-smoother",
                "color" : "#2ca02c",
                "line-width" : line_width,
                "line-style" : line_style
            },
            "energy" : {
                "name" : r"full-$\varphi_k$",
                "color" : "indigo",
                "line-width" : line_width,
                "line-style" : "--"
            }
        }
    }
    logging.debug("Adding `kalcal` plot options")

    plot_options["quartical"] = {
        "name" : r"QuartiCal",
        "label" : r"$\Delta_t$",
        "scale" : "linear",
        "color" : "#d62728",
        "line-width" : line_width,
        "line-style" : "-"
    }
    logging.debug("Adding `quartical` plot options")
    import scienceplots
    plt.style.use(["science", "no-latex"])
    logging.debug("`SciencePlots` module loaded, with latex, nature mode")

    logging.info("Updating parameter data")
    params["plots"] = plot_options
    params.save()

def place_grid(axes, axis="both"):
    logging.debug("Invoking function")
    for ax in axes:
        ax.grid(axis=axis, which="major", linewidth=0.8, alpha=0.6)
        ax.grid(axis=axis, which="minor", linewidth=0.6, alpha=0.2)
    
def plot_line(params, x_values, y_values, axis, label, style=None, 
                  marker="o", linewidth=None, near=0.0, 
                  xscale="linear", yscale="linear", color=None):
    logging.debug("Invoking function")

    plot_options = params["plots"]

    linewidth = plot_options["line-width"] if linewidth == None else linewidth
    style = plot_options["line-style"] if style == None else style
    x_values = np.log10(x_values) if xscale == "log" else x_values
    y_values = np.log10(y_values) if yscale == "log" else y_values
    
    logging.info(f"Plotting line: ({xscale}, {yscale}), " \
                + f"line-width={linewidth}, line-style={style}")
    
    if color == None:
        line, = axis.plot(x_values, y_values, ls=style, label=label, lw=linewidth)
        color = line.get_color()
    else:
        line, = axis.plot(x_values, y_values, ls=style, label=label, lw=linewidth, color=color)
    
    color_out = adjust_brightness(color, 0.6)
    index = np.abs(near - y_values).argmin()
    min_x, min_y = x_values[index], y_values[index]
    axis.plot(min_x, min_y, linestyle="", marker=marker, markerfacecolor=color,
                markersize=6, markeredgecolor=color_out)
    return line


def clip_axis(values, axis, adjust=(0.1, 0.85), mode="normal", scale="linear"):
    logging.debug("Invoking function")
    values = list(map(np.log10, values)) if scale == "log" else values
    data = np.sort(np.hstack([value.ravel() for value in values]).ravel())
    lb, ub = axis.get_ylim()
    
    if mode == "normal":
        lp, up = adjust
        lb = lp * data.min()
        ui = int(np.ceil(up * data.size))
        ui = (data.size - 1) if ui >= data.size else ui
        ub = data[ui]
    elif mode == "percentile":
        lp, up = adjust
        li = int(np.floor(lp * data.size))
        ui = int(np.ceil(up * data.size))
        
        li = 0 if li < 0 else li
        ui = data.size if ui >= data.size else ui
        
        lb, ub = data[li], data[ui]
    elif mode == "explicit":   
        lb, ub = adjust
    else:
        ValueError(f"Unknown mode: {mode} (Requires: normal, explicit)")
    
    axis.set_ylim((lb, ub))


def offset_axis(values, axis, offset):
    logging.debug("Invoking function")
    for value in values:
        value /= 10**offset
        
    axis.set_ylabel(axis.get_ylabel() + rf" $(\times 10^{{{offset}}})$")
    
    return values

def place_legend(params, lines, figure):
    logging.debug("Invoking function")
    label_size = params["plots"]["label-size"]
    mid = (figure.subplotpars.left + figure.subplotpars.right)/2
    labels = [line.get_label() for line in lines]
    ncol = len(labels)
    logging.debug(f"Placing legend: ({mid}, 0.0) with n-col={ncol}")
    figure.legend(lines, labels,
           loc='lower center', ncol=ncol, 
           frameon=True, prop={"size" : label_size},
           bbox_to_anchor=[mid, 0.0], 
           bbox_transform=figure.transFigure)

    
def place_markers(params, axes, adjust=(0.05, 0.1)):
    logging.debug("Invoking function")
    label_size = params["plots"]["label-size"]
    percents = params["percents"]
    lp, up = adjust
    
    for ax, percent in zip(axes, percents):
        xl, xu = ax.get_xlim()
        yl, yu = ax.get_ylim()
        dx = (xu - xl) * lp
        dy = (yu - yl) * up
        x = xl + dx
        y = yu - dy
        ax.text(x, y, f"{percent} MP",
                size=label_size,
                bbox=dict(facecolor='white', edgecolor='black', pad=3.0))

def get_min(x, y):
    logging.debug("Invoking function")
    return y[x.argmin()]

# def place_ticks(axes, tick_type=None, axis="x", scale="log"):
#     logging.debug("Invoking function")
#     for ax in axes:
#         if axis == "x" or axis == "both":
#             if isinstance(tick_type, (int, float)):
#                 ax.xaxis.set_major_locator(MultipleLocator(tick_type))
#                 ax.xaxis.set_minor_locator(MultipleLocator(tick_type / 5))
#             elif tick_type == None:
#                 ax.xaxis.set_major_locator(AutoLocator())
#                 ax.xaxis.set_minor_locator(AutoMinorLocator())
#             elif "kalcal" in tick_type:
#                 ax.xaxis.set_major_locator(MultipleLocator(0.5))
#                 ax.xaxis.set_major_formatter("{x:.1f}")     
#                 ax.xaxis.set_minor_locator(MultipleLocator(0.1))
#             elif "quartical" in tick_type:
#                 ax.xaxis.set_major_locator(MultipleLocator(40))
#                 ax.xaxis.set_major_formatter("{x:.0f}")
#                 ax.xaxis.set_minor_locator(MultipleLocator(8))
#             else:
#                 ax.xaxis.set_major_locator(AutoLocator())
#                 ax.xaxis.set_minor_locator(AutoMinorLocator())
#         if axis == "y" or axis == "both":
#             if isinstance(tick_type, (int, float)):
#                 ax.yaxis.set_major_locator(MultipleLocator(tick_type))
#                 ax.yaxis.set_minor_locator(MultipleLocator(tick_type / 5))
#             elif tick_type == None:
#                 ax.yaxis.set_major_locator(AutoLocator())
#                 ax.yaxis.set_minor_locator(AutoMinorLocator())
#             elif "kalcal" in tick_type:
#                 ax.yaxis.set_major_locator(MultipleLocator(0.5))
#                 ax.yaxis.set_major_formatter("{x:.1f}")     
#                 ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#             elif "quartical" in tick_type:
#                 ax.yaxis.set_major_locator(MultipleLocator(40))
#                 ax.yaxis.set_major_formatter("{x:.0f}")
#                 ax.yaxis.set_minor_locator(MultipleLocator(8))
#             else:
#                 ax.yaxis.set_major_locator(AutoLocator())
#                 ax.yaxis.set_minor_locator(AutoMinorLocator())

def place_ticks(axes, bounds=None, n_ticks=6, 
                stype="log", axis="x", precision=0):
    logging.debug("Invoking function")
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        if axis == "x" or axis == "both":                
            if stype == "log":
                lb, ub = bounds
                lb, ub = int(np.floor(lb)), int(np.ceil(ub))
                major_ticks = FixedLocator(np.arange(lb, ub + 0.5, 0.5))
                minor_ticks = AutoMinorLocator(5)
            elif stype == "linear":
                lb, ub = bounds
                major_step = round((ub - lb)/n_ticks, precision)
                major_ticks = MultipleLocator(base=major_step)
                minor_ticks = AutoMinorLocator(5)
            else:                
                major_ticks = AutoLocator()
                minor_ticks = AutoMinorLocator(5)
                
            ax.xaxis.set_major_locator(major_ticks)
            ax.xaxis.set_minor_locator(minor_ticks)
        
        if axis == "y" or axis == "both":
            if stype == "log":
                lb, ub = bounds
                lb, ub = int(np.floor(lb)), int(np.ceil(ub))
                major_ticks = FixedLocator(np.arange(lb, ub + 0.5, 0.5))
                minor_ticks = AutoMinorLocator(5)
            elif stype == "linear":
                lb, ub = bounds
                major_step = round((ub - lb)/n_ticks, precision)
                major_ticks = MultipleLocator(base=major_step)
                minor_ticks = AutoMinorLocator(5)
            else:
                major_ticks = AutoLocator()
                minor_ticks = AutoMinorLocator(5)

            ax.yaxis.set_major_locator(major_ticks)
            ax.yaxis.set_minor_locator(minor_ticks)
    

def init_plot(params, title, algorithms="", layout=(2, 2), size=None):
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
    figsize = (cm2in(plot_size * n_cols), cm2in(plot_size * n_rows))
    logging.debug(f"Subplots size: {figsize[0]}x{figsize[1]}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    try:
        axes = axes.ravel()
    except:
        pass
    quart_axes = []
    energy_axes = []
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
            ax.set_xscale("linear")
        if "quartical" in algorithms:
            logging.debug(f"Setup `quartical` twin y-axis={i}")
            ax2 = ax.twiny()
            ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
            ax2.set_xlabel(plot_options["quartical"]["label"], size=label_size)
            ax2.set_xscale(plot_options["quartical"]["scale"])
            ax2.invert_xaxis()
            ax2.tick_params(axis='x', which="both", labelsize=tick_size)
            quart_axes.append(ax2)
        if "energy" in algorithms:
            logging.debug(f"Setup `energy` twin x-axis={i}")
            ax2 = ax.twinx()
            ax2.set_xscale("linear")
            ax2.set_ylabel(plot_options["kalcal"]["energy"]["label"], size=label_size)
            ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
            ax2.tick_params(axis='y', which="both", labelsize=tick_size)
            energy_axes.append(ax2) 
        if "time" in algorithms:
            logging.debug(f"Setup `time` axes={i}")
            ax.set_xlabel(r"Time Step, $k$", size=label_size)
            ax.set_xscale("linear")
        if "source" in algorithms:
            logging.debug(f"Setup `source` axes={i}")
            ax.set_xlabel("Source Index", size=label_size)
            ax.set_xscale("linear")
        else:
            logging.debug(f"Setup Custom")
    
    if ("kalcal" in algorithms and "quartical" in algorithms) \
    or ("kalcal" in algorithms and "energy" in algorithms):
        logging.debug(f"Adjust subplots for twin axes")       
        top = 0.8 if "\n" in title else 0.875
        hspace = 0.5
        wspace = 0.4
        right = 0.9
    else:
        logging.debug(f"Adjust subplots for single axes")
        top = 0.775 if "\n" in title else 0.9
        hspace = 0.4
        wspace = 0.4
        right = None

    if "time" in algorithms or "source" in algorithms:
        logging.debug("No legend adjust")
        bottom = 0.2
    else:
        bottom = 0.225

    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, 
                        wspace=wspace, right=right)

    logging.info(f"Figure and axes setup complete")
    return fig, axes, quart_axes, energy_axes


def create_amplitude_and_phase_signal_plot(ants, params, show=False,
                                            overwrite=False):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    if paths["plots"]["files"].get("ch5-sim-gains", False):
        plot_path = paths["plots"]["files"]["ch5-sim-gains"]
    else:
        plot_path = paths["plots"]["dir"] / "ch5-sim-gains.png"

    if not overwrite: 
        try:
            logging.debug("Checking for existing plot")
            check_for_data(plot_path)
        except:
            logging.info("No plotting done")
            return
    else:
        logging.debug("Overwriting plot")

    logging.info("Load the true-gains data")
    true_path = paths["gains"]["true"]["files"]
    gains = load_data(true_path)
    
    amp_gains = gains["amp_gains"]
    phase_gains = gains["phase_gains"]        
    n_rows, n_cols = len(ants), 4

    logging.info("Creating plot")
    fig, axes, _, _ = init_plot(
        params,
        title="Components of the True Gains Signal",
        algorithms="time",
        layout=(n_rows, n_cols),
        size=9
    )
    
    label_size = plot_options["label-size"]
    line_width = plot_options["line-width"]
    line_style = plot_options["line-style"]
    n_time = amp_gains.shape[0]
    times = np.arange(1, n_time + 1)
    axes[0].set_title(r"Amplitude, $\mathbf{A}(t)$")
    axes[1].set_title(r"Phase, $\mathbf{\phi}(t)$, in radians")
    axes[2].set_title(r"Real Part, $\Re(\mathbf{g}(t))$")
    axes[3].set_title(r"Imaginary Part, $\Im(\mathbf{g}(t))$")

    logging.info("Plotting amplitude and phase signals")
    for i, ant in enumerate(ants):       
        c = 4 * i

        logging.debug(f"Plot amplitude over time for `ant={ant}`")
        apq = amp_gains[:, ant]
        axes[c].plot(times, apq.real, color="red", 
                     ls=line_style, lw=line_width)
        axes[c].set_ylabel(rf"$A_{{{ant + 1}}}(t)$", size=label_size)
        
        logging.debug(f"Plot phase over time for `ant={ant}`")
        ppq = phase_gains[:, ant]
        axes[c + 1].plot(times, ppq.real, color="green", 
                     ls=line_style, lw=line_width)
        axes[c + 1].set_ylabel(rf"$\phi_{{{ant + 1}}}(t)$", size=label_size)

        tpq = apq.real * np.exp(1.0j * ppq.real)
        axes[c + 2].plot(times, tpq.real, color="blue", 
                     ls=line_style, lw=line_width)
        axes[c + 2].set_ylabel(rf"$\Re(g_{{{ant + 1}}}(t))$", size=label_size)
        axes[c + 3].plot(times, tpq.imag, color="darkorange", 
                     ls=line_style, lw=line_width)
        axes[c + 3].set_ylabel(rf"$\Im(g_{{{ant + 1}}}(t))$", size=label_size)

    # Place ticks and grid 
    place_ticks(axes, (1, n_time), 6, stype="linear")
    place_grid(axes)

    logging.debug("Placing ticks")

    # Adjust subplots
    fig.subplots_adjust(
        top=0.93, hspace=0.25, wspace=0.45
    )

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])

    paths["plots"]["files"]["ch5-sim-gains"] = plot_path
    params.save()

    if show:
        logging.info("Displaying amplitude and phase signal plot")
        plt.show()


def create_source_distribution_plot(params, show=False, overwrite=False):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    if paths["plots"]["files"].get("ch5-sim-skymodel", False):
        plot_path = paths["plots"]["files"]["ch5-sim-skymodel"]
    else:
        plot_path = paths["plots"]["dir"] / "ch5-sim-skymodel.png"
    
    if not overwrite:
        try:
            logging.debug("Checking for existing plot")
            check_for_data(plot_path)
        except:
            logging.info("No plotting done")
            return
    else:
        logging.info("Overwriting plot")

    title_size= plot_options["title-size"]
    label_size = plot_options["label-size"]
    cmap = plt.cm.get_cmap('plasma')

    percents = params["percents"]
    n_rows, n_cols = len(percents), 2

    true_paths = paths["fluxes"]["true"]["files"]
    skymodel = load_data(true_paths[100])
    true_model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    true_flux = true_model[Ix, Iy]
    n_src = len(true_flux)
    n_pix = true_model.shape[0]
    params["n-src"] = n_src
    params["n-pix"] = n_pix
    params.save()

    # Plot image for each percent
    true_flux = np.clip(true_flux, a_min=1e-6, a_max=None)
    log_flux = np.log10(true_flux)
    std_flux = (log_flux - log_flux.min()) / np.ptp(log_flux)
    # std_flux = (true_flux - true_flux.min())/np.ptp(true_flux)
    colors = cmap(std_flux)
    sources = np.arange(n_src) + 1
    radius = 5/480 * n_pix

    logging.info("Creating plot")
    fig, axes, _, _ = init_plot(
        params,
        title="Simulated Models of the Sky\nover various Modelled Flux Percentage",
        algorithms="source",
        layout=(n_cols, n_rows),
        size=14
    )    

    axes = axes.reshape(n_cols, n_rows).transpose()

    # Plot each image
    for i, percent in enumerate(percents):

        # Plot bar
        model = load_data(true_paths[percent])["model"]
        sub_flux = model[Ix, Iy]
        srcs = len(np.nonzero(sub_flux)[0])        
        axes[i, 0].set_title(f"{percent} MP", size=title_size)
        axes[i, 1].set_title(f"{srcs} sources", size=title_size)  
            
        # Plot image
        axes[i, 1].grid(False)
        model[model == 0] = 1e-7
        im = axes[i, 1].imshow(model, cmap=cmap, norm=mpl.colors.LogNorm())    
        axes[i, 1].set_xlabel(rf"$N_x = {n_pix}$", size=label_size)
        axes[i, 1].set_ylabel(rf"$N_y = {n_pix}$", size=label_size)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    
        # Plot circles around each point to indicate position
        rev_coords = list(enumerate(zip(Ix, Iy)))[::-1]
        for j, (x, y) in rev_coords:
            if model[x, y] == 1e-7:
                continue
            circle = plt.Circle((y, x), radius, color=colors[j], fill=True)
            axes[i, 1].add_patch(circle)
        
        axes[i, 0].bar(sources, sub_flux, color=colors, width=1)
        axes[i, 0].set_xlabel("Source Index", size=label_size)
        axes[i, 0].set_ylabel("Jy/pixel", size=label_size)
    
    fig.subplots_adjust(
        top=0.87, hspace=0.3, wspace=0.4
    )

    cb = fig.colorbar(im, ax=axes.flatten(), location="right")
    cb.set_label(label="Jy/pixel", size=label_size)

    mid = 5.1 * (fig.subplotpars.right + fig.subplotpars.left)/12
    fig.suptitle("Simulated Models of the Sky\nover various Modelled Flux Percentage", 
                 size=title_size, x=mid) 

    place_grid(axes[:, 0])

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])

    paths["plots"]["files"]["ch5-sim-skymodel"] = plot_path
    params.save()

    if show:
        logging.info("Source distribution plot")
        plt.show()

def create_fits_grid_plot(params, algorithm, indices, labels, bounds, itype="residual", show=False, overwrite=False):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    name = f"ch5-fits-grid-{algorithm}-{itype}-diff"

    if paths["plots"]["files"].get(name, False):
        plot_path = paths["plots"]["files"][name]
    else:
        plot_path = paths["plots"]["dir"] / f"{name}.png"
    
    if not overwrite:
        try:
            logging.debug("Checking for existing plot")
            check_for_data(plot_path)
        except:
            logging.info("No plotting done")
            return
    else:
        logging.info("Overwriting plot")

    title_size= plot_options["title-size"]
    label_size = plot_options["label-size"]
    cmap = plt.cm.get_cmap('plasma')

    percents = params["percents"]
    n_rows, n_cols = len(percents), 2

    true_models = paths["fluxes"]["true"]["files"]
    skymodel = load_data(true_models[100])
    true_model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    coords = list(zip(Ix, Iy))
    true_flux = true_model[Ix, Iy]
    n_src = len(true_flux)
    n_pix = true_model.shape[0]
    params["n-src"] = n_src
    params["n-pix"] = n_pix
    params.save()

    logging.info("Load the true-gains data")
    true_paths = paths["fluxes"]["true"]["files"]["solved"] 
    if "kalcal" in algorithm.lower():
        alg_paths = paths["fluxes"][algorithm.lower()]["smoother"]["files"]
        param_values = params[algorithm.lower()]["sigma-fs"]
        y_label = plot_options["kalcal"]["label"]
    else:
        alg_paths = paths["fluxes"][algorithm.lower()]["files"]
        param_values = params[algorithm.lower()]["t-ints"]
        y_label = plot_options[algorithm.lower()]["label"]

    indices = np.array(indices)
    n_rows, n_cols = indices.shape

    logging.info("Creating plot")
    fig, axes, _, _ = init_plot(
        params,
        title=f"Difference between {itype.capitalize()} Images:\n{algorithm} against Perfect Solutions",
        algorithms="",
        layout=(n_rows, n_cols),
        size=9
    )

    vmin, vmax = bounds
    axes = axes.reshape(n_rows, n_cols)
    radius = 40/480 * n_pix
        
    logging.info("Creating grid")
    true_image = load_data(true_paths[100])[itype.lower()]
    true_model = load_data(true_models[percents[-1]])["model"]
    for i in range(indices.shape[0]):
        param_label = labels[i]
        for j, percent in enumerate(percents):
            idx = indices[i, j]
            param_value = param_values[idx]

            if "kalcal" in algorithm:
                display_value = f"{np.log10(param_value):.2g}"
            else:
                display_value = param_value

            axes[i, j].set_xlabel(f"{y_label} = {display_value}", size=label_size)
            if i == 0:
                axes[i, j].set_title(f"{percent} MP", size=title_size)
            if j == 0:
                axes[i, j].set_ylabel(param_label, size=label_size)

            path = alg_paths[percent][param_value]
            image = load_data(path)[itype.lower()]
            image -= true_image
            mat = axes[i, j].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for x, y in coords:
                if true_model[x, y] == 0.0:
                    break
                circle = plt.Rectangle((y - radius/2, x - radius/2), radius, radius, color="white", fill=False, alpha=0.6)
                axes[i, j].add_patch(circle)

    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])
    cb = fig.colorbar(mat, cax=cbar_ax)
    cb.set_label(label=r"Flux Per Beam, Jy/beam", size=label_size, labelpad=20)

    # Adjust subplots
    fig.subplots_adjust(
        top=0.93, hspace=0.2, wspace=0.05
    )

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])

    paths["plots"]["files"][name] = plot_path
    params.save()

    if show:
        logging.info(f"Displaying {itype} grid plot")
        plt.show()

def create_flux_ratio_per_source_plot(params, scale="Major", show=False, overwrite=False):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    name = f"ch5-flux-ratio-per-src-{scale.lower()}"
    if paths["plots"]["files"].get(name, False):
        plot_path = paths["plots"]["files"][name]
    else:
        plot_path = paths["plots"]["dir"] / f"{name}.png"
    
    if not overwrite:
        try:
            logging.debug("Checking for existing plot")
            check_for_data(plot_path)
        except:
            logging.info("No plotting done")
            return
    else:
        logging.info("Overwriting plot")

    title_size= plot_options["title-size"]
    label_size = plot_options["label-size"]
    tick_size = plot_options["tick-size"]
    cmap = plt.cm.get_cmap('plasma')

    percents = params["percents"]
    n_rows, n_cols = 3, len(percents)

    true_paths = paths["fluxes"]["true"]["files"]
    skymodel = load_data(true_paths[100])
    true_model = skymodel["model"]
    Ix = skymodel["Ix"]
    Iy = skymodel["Iy"]
    true_flux = true_model[Ix, Iy]
    n_src = len(true_flux)
    params["n-src"] = n_src
    params.save()
    size = cm2in(14)
    diag_points = params["kalcal-diag"]["n-points"]
    full_points = params["kalcal-full"]["n-points"]
    quart_points = params["quartical"]["n-points"]
    min_points = min(diag_points,
                     full_points,
                     quart_points)
    r = min_points/n_src
    logging.info("Creating plot")
    fig, axes = plt.subplots(3, len(percents), 
                             figsize=(4 * size, 3 * r * size))
    fig.subplots_adjust(
            top=0.85,
            bottom=0.125,
            hspace=0.25, 
            wspace=0.35,
            right=None
    )    
    axes = axes.reshape(n_rows, n_cols)

    cmap1 = plt.cm.get_cmap("Reds")
    cmap2 = plt.cm.get_cmap("Blues_r")
    tophalf = np.zeros((256, 4))
    bothalf = np.zeros((256, 4))

    for i in np.arange(256):
        tophalf[i] = cmap1(i)
        bothalf[i] = cmap2(i)
        
    all_colors = np.vstack((bothalf, tophalf))
    cmap = LinearSegmentedColormap.from_list('two_slope_cmap', all_colors)
    if scale == "Major":
        vmin = 0.5
        vcenter = 1.0
        vmax = 1.5
    else:
        vmin = 0.9
        vcenter = 1.0
        vmax = 1.1

    divnorm = TwoSlopeNorm(vmin=vmin - (vcenter - vmin)/127, vcenter=vcenter, vmax=vmax)

    diag_sigma_fs = params["kalcal-diag"]["sigma-fs"]
    full_sigma_fs = params["kalcal-full"]["sigma-fs"]
    t_ints = params["quartical"]["t-ints"]
    
    for ax in axes[-1]:
        ax.set_xlabel(rf"Source Index", size=tick_size)

    # Plot for each percent across params
    for i, percent in enumerate(percents):    
        # Set title
        axes[0, i].set_title(f"{percent} MP", size=title_size)
        
        diag_kalcal_ratio = np.zeros((len(diag_sigma_fs), len(true_flux)), dtype=np.float64)
        full_kalcal_ratio = np.zeros((len(full_sigma_fs), len(true_flux)), dtype=np.float64)
        quartical_ratio = np.zeros((len(t_ints), len(true_flux)), dtype=np.float64)
        
        for j, path in enumerate(paths["fluxes"]["kalcal-diag"]["smoother"]["files"][percent].values()):
            recovered_flux = load_data(path)["flux"]
            diag_kalcal_ratio[j] = flux_ratio(recovered_flux, true_flux)
        
        for j, path in enumerate(paths["fluxes"]["kalcal-full"]["smoother"]["files"][percent].values()):
            recovered_flux = load_data(path)["flux"]
            full_kalcal_ratio[j] = flux_ratio(recovered_flux, true_flux)
            
        for j, path in enumerate(paths["fluxes"]["quartical"]["files"][percent].values()):
            recovered_flux = load_data(path)["flux"]
            quartical_ratio[j] = flux_ratio(recovered_flux, true_flux)

        if percent == 100:
            axes[0, i].text(-36, diag_points//6, "kalcal-diag", size=label_size, rotation=90,
                    bbox=dict(facecolor='none', edgecolor='black', pad=3.0))
            
        axes[0, i].set_ylabel(plot_options["kalcal"]["label"], size=tick_size)  
        axes[0, i].set_yticks([0, diag_points//2 - 1, diag_points - 1], 
                            [f'{params["kalcal-diag"]["low-bound"]}', 
                            f'{(params["kalcal-diag"]["low-bound"] + params["kalcal-diag"]["up-bound"])//2}',
                            f'{params["kalcal-diag"]["up-bound"]}'], size=tick_size)
        mat = axes[0, i].imshow(diag_kalcal_ratio, cmap=cmap, norm=divnorm, interpolation='nearest', aspect='auto')
        axes[0, i].invert_yaxis()

        if percent == 100:
            axes[1, i].text(-36, full_points//6, "kalcal-full", size=label_size, rotation=90,
                    bbox=dict(facecolor='none', edgecolor='black', pad=3.0))
            
        axes[1, i].set_ylabel(plot_options["kalcal"]["label"], size=tick_size)  
        axes[1, i].set_yticks([0, full_points//2 - 1, full_points - 1], 
                            [f'{params["kalcal-full"]["low-bound"]}', 
                            f'{(params["kalcal-full"]["low-bound"] + params["kalcal-full"]["up-bound"])//2}',
                            f'{params["kalcal-full"]["up-bound"]}'], size=tick_size)
        mat = axes[1, i].imshow(full_kalcal_ratio, cmap=cmap, norm=divnorm, interpolation='nearest', aspect='auto') 
        axes[1, i].invert_yaxis()  
        
        if percent == 100:        
            axes[2, i].text(-36, 5*quart_points//6, "QuartiCal", size=label_size, rotation=90,
                    bbox=dict(facecolor='none', edgecolor='black', pad=3.0))
            
        axes[2, i].set_ylabel(plot_options["quartical"]["label"], size=tick_size)
        axes[2, i].set_yticks([0, quart_points//2 - 1, quart_points - 1], 
                              t_ints[[0, quart_points//2 - 1, quart_points - 1]], size=tick_size)
        
        mat = axes[2, i].imshow(quartical_ratio, cmap=cmap, norm=divnorm, interpolation='nearest', aspect='auto')

    cb = fig.colorbar(mat, ax=axes.ravel().tolist())
    cb.set_label(label=r"Flux Ratio Per Source, $\gamma_i$", size=label_size, labelpad=10)

    mid = 5.1 * (fig.subplotpars.right + fig.subplotpars.left)/12
    fig.suptitle(f"({scale}) Flux Ratio of Recovered Flux per Source:\nkalcal vs QuartiCal", size=title_size, x=mid)    

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])

    paths["plots"]["files"][name] = plot_path
    params.save()

    if show:
        logging.info(f"({scale}) Flux ratio per source plot")
        plt.show()


def create_standard_plot(params, **kwargs):
    logging.debug("Invoking function")
    paths = params["paths"]
    plot_options = params["plots"]
    percents = params["percents"]

    if params["seed"]:
        logging.info(f"Setting seed to {params['seed']}")
        np.random.seed(params["seed"])
    else:
        logging.info(f"No seed set")

    logging.debug("Get plot details")
    title = kwargs["title"]
    metric = kwargs["metric"]
    name = kwargs["name"]
    algorithms = kwargs["algorithms"]
    size = kwargs["size"]
    subplot_adjust = kwargs["subplot_adjust"]
    limit_adjust = kwargs["limit_adjust"]
    offset = kwargs.get("offset", None)

    if paths["plots"]["files"].get(name, False):
        plot_path = paths["plots"]["files"][name]
    else:
        plot_path = paths["plots"]["dir"] / f"{name}.png"
    
    if not kwargs.get("overwrite", True):
        try:
            logging.debug("Checking for existing plot")
            check_for_data(plot_path)
        except:
            logging.info("No plotting done")
            return
    else:
        logging.info("Overwriting previous plot")
        
    algorithm = []
    subheading = []
    suffix = []
    
    diag_smoother_status = True if "diag-smoother" in algorithms else False
    diag_filter_status = True if "diag-filter" in algorithms else False
    diag_energy_status = True if "diag-energy" in algorithms else False
    full_smoother_status = True if "full-smoother" in algorithms else False
    full_filter_status = True if "full-filter" in algorithms else False
    full_energy_status = True if "full-energy" in algorithms else False
    quartical_status = True if "quartical" in algorithms else False

    kalcal_diag_status = (diag_smoother_status or diag_filter_status or diag_energy_status)
    kalcal_full_status = (full_smoother_status or full_filter_status or full_energy_status)

    if kalcal_diag_status or kalcal_full_status:
        algorithm.append("kalcal")
        if quartical_status:
            subheading.append("kalcal")
        else:
            if kalcal_diag_status:
                subheading.append("kalcal-diag")
            if kalcal_full_status:
                subheading.append("kalcal-full")

            if diag_filter_status or full_filter_status:
                suffix.append("filter")
            if diag_energy_status or full_energy_status:
                suffix.append("energy function")
    
    if quartical_status:
        algorithm.append("quartical")
        subheading.append("QuartiCal")

    if diag_filter_status or full_filter_status:
        algorithm.append("filter")

    if diag_energy_status or full_energy_status:
        algorithm.append("energy")

    if len(algorithm) or len(subheading):
        algorithm = "+".join(algorithm)
        subheading = " vs ".join(subheading)
        if len(suffix):
            suffix = " and ".join(suffix)
            subheading += f", with {suffix}"
    else:
        logging.info("All algorithms are disabled, do nothing")
        return

    metric_function, y_label, m_type = {
        "gains-mse": (calculate_mse_on_gains, "MSE", "gains"),
        "flux-mse" : (calculate_mse_on_fluxes, "MSE", "fluxes"),
        "flux-ratio" : (calculate_total_ratio_on_fluxes, r"$\gamma$", "fluxes"),
        "flux-psi-0" : (calculate_psi_0_on_fluxes, r"$\psi_0$", "fluxes"),
        "flux-psi-1": (calculate_psi_1_on_fluxes, r"$\psi_1$", "fluxes"),
        "rms-on-residuals" : (calculate_rms_from_residuals, "RMS", "fluxes"),
        "rms-on-corrected" : (calculate_rms_from_corrected_residuals, "RMS", "fluxes")
    }[metric.lower()]

    if metric_function == calculate_total_ratio_on_fluxes:
        near = 1.0
    else:
        near = 0.0

    n_plots = len(percents)
    if n_plots % 2:
        layout = (1, n_plots)
    else:
        n_rows = int(np.floor(np.sqrt(n_plots)))
        n_cols = n_plots // n_rows
        layout = (n_rows, n_cols)

    logging.info("Creating plot")
    fig, axes, quart_axes, energy_axes = init_plot(
        params,
        title=f"{title}:\n{subheading}",
        algorithms=algorithm,
        size=size,
        layout=layout
    )    

    lines = []
    values = {}
    label_size = plot_options["label-size"]
    
    if int(near):
        for i in range(len(percents)):
            line = axes[i].axhline(y=1.0, color="black", lw=2.0, label="Unity")
        lines.append(line)

    if diag_smoother_status:
        if diag_filter_status or diag_energy_status or full_filter_status or full_energy_status:
             diag_label = plot_options["kalcal"]["diagonal"]["smoother"]["name"]
        else:
             diag_label = plot_options["kalcal"]["diagonal"]["name"]
        
        diag_color = plot_options["kalcal"]["diagonal"]["smoother"]["color"]
        diag_sigma_fs = params["kalcal-diag"]["sigma-fs"]
        diag_width = plot_options["kalcal"]["diagonal"]["smoother"]["line-width"]
        diag_style = plot_options["kalcal"]["diagonal"]["smoother"]["line-style"]
        diag_scale = plot_options["kalcal"]["scale"]
        diag_paths = paths[m_type]["kalcal-diag"]["smoother"]["files"]
        values["diag-smoother"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            metric_values = metric_function(params, diag_paths[percent])
            optimums = np.abs(near - metric_values)
            opt_idx = optimums.argmin()
            min_var = np.square(optimums - optimums.min()).mean()
            logging.info(f"kalcal-diag smoother on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {diag_sigma_fs[opt_idx]:.3g} (10^({np.log10(diag_sigma_fs[opt_idx]):.3g}))"\
                        + f" --> {metric_values[opt_idx]:.3g} (+- {min_var:.3g})")
            output_str += [
                f"{diag_sigma_fs[opt_idx]:.3g}", f"{metric_values[opt_idx]:.3g}", f"{min_var:.3g}"  
            ]
            if offset != None:
                metric_values /= 10**offset
            values["diag-smoother"][percent] = metric_values
            line = plot_line(params, diag_sigma_fs, metric_values, axes[i], 
                      label=diag_label, style=diag_style, 
                      linewidth=diag_width, xscale=diag_scale,
                      color=diag_color, near=near)
        
        logging.info(f" & ".join(output_str) + " \\\\")        
        lines.append(line)

    if diag_filter_status:
        diag_label = plot_options["kalcal"]["diagonal"]["filter"]["name"]        
        diag_color = plot_options["kalcal"]["diagonal"]["filter"]["color"]
        diag_sigma_fs = params["kalcal-diag"]["sigma-fs"]
        diag_width = plot_options["kalcal"]["diagonal"]["filter"]["line-width"]
        diag_style = plot_options["kalcal"]["diagonal"]["filter"]["line-style"]
        diag_scale = plot_options["kalcal"]["scale"]
        diag_paths = paths[m_type]["kalcal-diag"]["filter"]["files"]
        values["diag-filter"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            metric_values = metric_function(params, diag_paths[percent])
            optimums = np.abs(near - metric_values)
            opt_idx = optimums.argmin()
            min_var = np.square(optimums - optimums.min()).mean()
            logging.info(f"kalcal-diag filter on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {diag_sigma_fs[opt_idx]:.3g} (10^({np.log10(diag_sigma_fs[opt_idx]):.3g}))"\
                        + f" --> {metric_values[opt_idx]:.3g} (+- {min_var:.3g})")
            output_str += [
                f"{diag_sigma_fs[opt_idx]:.3g}", f"{metric_values[opt_idx]:.3g}", f"{min_var:.3g}"  
            ]
            if offset != None:
                metric_values /= 10**offset
            values["diag-filter"][percent] = metric_values
            line = plot_line(params, diag_sigma_fs, metric_values, axes[i], 
                      label=diag_label, style=diag_style, 
                      linewidth=diag_width, xscale=diag_scale,
                      color=diag_color, near=near)
        
        logging.info(f" & ".join(output_str) + " \\\\")         
        lines.append(line)

    if diag_energy_status:
        diag_color = plot_options["kalcal"]["diagonal"]["energy"]["color"]
        diag_sigma_fs = params["kalcal-diag"]["sigma-fs"]
        diag_label = plot_options["kalcal"]["diagonal"]["energy"]["name"]
        diag_width = plot_options["kalcal"]["diagonal"]["energy"]["line-width"]
        diag_style = plot_options["kalcal"]["diagonal"]["energy"]["line-style"]
        diag_scale = plot_options["kalcal"]["scale"]
        diag_paths = paths["gains"]["kalcal-diag"]["filter"]["files"]
        values["diag-energy"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            energy_axes[i].set_ylabel(plot_options["kalcal"]["energy"]["label"],
                                        size=label_size)
            energy_values = get_energy_values(diag_paths[percent])
            opt_idx = energy_values.argmin()
            logging.info(f"kalcal-diag energy on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {diag_sigma_fs[opt_idx]:.3g} "\
                        + f"(10^({np.log10(diag_sigma_fs[opt_idx]):.3g})) = {energy_values.min():.3E}")
            values["diag-energy"][percent] = energy_values
            output_str += [
                f"{diag_sigma_fs[opt_idx]:.3g}", f"{energy_values[opt_idx]:.3g}", ""  
            ]
            if energy_values.max() >= 1e4:
                energy_offset = int(np.log10(energy_values.max()))
                energy_values, = offset_axis([energy_values], energy_axes[i], energy_offset)
            line = plot_line(params, diag_sigma_fs, energy_values, energy_axes[i], 
                      label=diag_label, style=diag_style, 
                      linewidth=diag_width, xscale=diag_scale,
                      yscale="linear", color=diag_color, near=near)

            min_val = energy_values.min()
            min_var = np.sqrt(np.square(energy_values - energy_values.min()).mean())
            logging.info(f"Energy function on {percent}MP: Min=({min_val:.3f}), Range=({min_val - min_var:.3f}, {min_val + min_var:.3f})")
        logging.info(f" & ".join(output_str) + " \\\\")          
        lines.append(line)

    if full_smoother_status:
        if diag_filter_status or diag_energy_status or full_filter_status or full_energy_status:
             full_label = plot_options["kalcal"]["full"]["smoother"]["name"]
        else:
             full_label = plot_options["kalcal"]["full"]["name"]
        
        full_color = plot_options["kalcal"]["full"]["smoother"]["color"]
        full_sigma_fs = params["kalcal-full"]["sigma-fs"]
        full_width = plot_options["kalcal"]["full"]["smoother"]["line-width"]
        full_style = plot_options["kalcal"]["full"]["smoother"]["line-style"]
        full_scale = plot_options["kalcal"]["scale"]
        full_paths = paths[m_type]["kalcal-full"]["smoother"]["files"]
        values["full-smoother"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            metric_values = metric_function(params, full_paths[percent])
            optimums = np.abs(near - metric_values)
            opt_idx = optimums.argmin()
            min_var = np.square(optimums - optimums.min()).mean()
            logging.info(f"kalcal-full smoother on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {full_sigma_fs[opt_idx]:.3g} (10^({np.log10(full_sigma_fs[opt_idx]):.3g}))"\
                        + f" --> {metric_values[opt_idx]:.3g} (+- {min_var:.3g})")
            output_str += [
                f"{full_sigma_fs[opt_idx]:.3g}", f"{metric_values[opt_idx]:.3g}", f"{min_var:.3g}"  
            ]
            if offset != None:
                metric_values /= 10**offset
            values["full-smoother"][percent] = metric_values
            line = plot_line(params, full_sigma_fs, metric_values, axes[i], 
                      label=full_label, style=full_style, 
                      linewidth=full_width, xscale=full_scale,
                      color=full_color, near=near)
        
        logging.info(f" & ".join(output_str) + " \\\\")  
        lines.append(line)

    if full_filter_status:
        full_label = plot_options["kalcal"]["full"]["filter"]["name"]
        full_color = plot_options["kalcal"]["full"]["filter"]["color"]
        full_sigma_fs = params["kalcal-full"]["sigma-fs"]
        full_width = plot_options["kalcal"]["full"]["filter"]["line-width"]
        full_style = plot_options["kalcal"]["full"]["filter"]["line-style"]
        full_scale = plot_options["kalcal"]["scale"]
        full_paths = paths[m_type]["kalcal-full"]["filter"]["files"]
        values["full-filter"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            metric_values = metric_function(params, full_paths[percent])
            optimums = np.abs(near - metric_values)
            opt_idx = optimums.argmin()
            min_var = np.square(optimums - optimums.min()).mean()
            logging.info(f"kalcal-full filter on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {full_sigma_fs[opt_idx]:.3g} (10^({np.log10(full_sigma_fs[opt_idx]):.3g}))"\
                        + f" --> {metric_values[opt_idx]:.3g} (+- {min_var:.3g})")
            output_str += [
                f"{full_sigma_fs[opt_idx]:.3g}", f"{metric_values[opt_idx]:.3g}", f"{min_var:.3g}"  
            ]
            if offset != None:
                metric_values /= 10**offset
            values["full-filter"][percent] = metric_values
            line = plot_line(params, full_sigma_fs, metric_values, axes[i], 
                      label=full_label, style=full_style, 
                      linewidth=full_width, xscale=full_scale,
                      color=full_color, near=near)
            
        logging.info(f" & ".join(output_str) + " \\\\")        
        lines.append(line)

    if full_energy_status:
        full_color = plot_options["kalcal"]["full"]["energy"]["color"]
        full_sigma_fs = params["kalcal-full"]["sigma-fs"]
        full_label = plot_options["kalcal"]["full"]["energy"]["name"]
        full_width = plot_options["kalcal"]["full"]["energy"]["line-width"]
        full_style = plot_options["kalcal"]["full"]["energy"]["line-style"]
        full_scale = plot_options["kalcal"]["scale"]
        full_paths = paths["gains"]["kalcal-full"]["filter"]["files"]
        values["full-energy"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            energy_axes[i].set_ylabel(plot_options["kalcal"]["energy"]["label"],
                                        size=label_size)
            energy_values = get_energy_values(full_paths[percent])
            opt_idx = energy_values.argmin()
            logging.info(f"kalcal-full energy on `{metric}` for {percent}MP: "\
                        + f"sigma-fs['{opt_idx}'] = {full_sigma_fs[opt_idx]:.3g} "\
                        + f"(10^({np.log10(full_sigma_fs[opt_idx]):.3g})) = {energy_values.min():.3E}")
            values["full-energy"][percent] = energy_values
            output_str += [
                f"{full_sigma_fs[opt_idx]:.3g}", f"{energy_values[opt_idx]:.3g}", ""  
            ]
            if energy_values.max() >= 1e4:
                energy_offset = int(np.log10(energy_values.max()))
                energy_values, = offset_axis([energy_values], energy_axes[i], energy_offset)
            line = plot_line(params, full_sigma_fs, energy_values, energy_axes[i], 
                      label=full_label, style=full_style, 
                      linewidth=full_width, xscale=full_scale,
                      yscale="log", color=full_color, near=near)
        
            min_val = energy_values.min()
            min_var = np.sqrt(np.square(energy_values - energy_values.min()).mean())
            logging.info(f"Energy function on {percent}MP: Min=({min_val:.3f}), Range=({min_val - min_var:.3f}, {min_val + min_var:.3f})")
        logging.info(f" & ".join(output_str) + " \\\\")
        lines.append(line)

    if quartical_status:
        quart_color = plot_options["quartical"]["color"]
        quart_t_ints = params["quartical"]["t-ints"]
        quart_label = plot_options["quartical"]["name"]
        quart_width = plot_options["quartical"]["line-width"]
        quart_style = plot_options["quartical"]["line-style"]
        quart_scale = plot_options["quartical"]["scale"]
        quart_paths = paths[m_type]["quartical"]["files"]
        values["quartical"] = {}

        # Plot each image
        output_str = [str(metric)]
        for i, percent in enumerate(percents):
            metric_values = metric_function(params, quart_paths[percent])
            optimums = np.abs(near - metric_values)
            opt_idx = optimums.argmin()
            min_var = np.square(optimums - optimums.min()).mean()
            logging.info(f"QuartiCal on `{metric}` for {percent}MP: "\
                        + f"t-ints['{opt_idx}'] = {quart_t_ints[opt_idx]}, "\
                        + f" --> {metric_values[opt_idx]:.3g} (+- {min_var:.3g})")
            output_str += [
                f"{quart_t_ints[opt_idx]:.3g}", f"{metric_values[opt_idx]:.3g}", f"{min_var:.3g}"  
            ]
            if offset != None:
                metric_values /= 10**offset
            values["quartical"][percent] = metric_values
            line = plot_line(params, quart_t_ints, metric_values, quart_axes[i], 
                      label=quart_label, style=quart_style, 
                      linewidth=quart_width, xscale=quart_scale,
                      color=quart_color, near=near)
                
        logging.info(f" & ".join(output_str) + " \\\\")
        lines.append(line)
    
    keys = values.keys()
    for i, percent in enumerate(percents):
        sub_values = []
        energy_values = []

        if "diag-smoother" in keys:
            sub_values += [values["diag-smoother"][percent]]
        if "diag-filter" in keys:
            sub_values += [values["diag-filter"][percent]]
        if "diag-energy" in keys:
            energy_values += [values["diag-energy"][percent]]
        if "full-smoother" in keys:
            sub_values += [values["full-smoother"][percent]]
        if "full-filter" in keys:
            sub_values += [values["full-filter"][percent]]
        if "full-energy" in keys:
            energy_values += [values["full-energy"][percent]]
        if "quartical" in keys:
            sub_values += [values["quartical"][percent]]
        
        if len(sub_values):
            mode = limit_adjust.get("mode", "normal")
            if isinstance(mode, dict):
                mode = mode.get("main", "normal")

            adjust = limit_adjust.get("adjust", (0.0, 0.85))

            if isinstance(adjust, dict):
                if adjust.get(percent, False):
                    adjust = adjust[percent]
                elif adjust.get("main", False):
                    if isinstance(adjust["main"], dict) and adjust["main"].get(percent, False):
                        adjust = adjust["main"][percent]
                    else:
                        adjust = adjust["main"]
                else:
                    adjust = (0.0, 0.85)
            
            clip_axis(sub_values, axes[i], mode=mode, adjust=adjust)
            axes[i].set_ylabel(y_label, size=label_size)
            if offset != None:
                offset_axis(sub_values, axes[i], offset)

        if len(energy_values):
            mode = limit_adjust.get("mode", "normal")
            if isinstance(mode, dict):
                mode = mode.get("energy", "normal")

            adjust = limit_adjust.get("adjust", (0.0, 0.85))
            if isinstance(adjust, dict):
                if adjust.get(percent, False):
                    adjust = adjust[percent]
                elif adjust.get("energy", False):
                    if isinstance(adjust["energy"], dict) and adjust["energy"].get(percent, False):
                        adjust = adjust["energy"][percent]
                    else:
                        adjust = adjust["energy"]
                else:
                    adjust = (0.0, 0.85)
        
            clip_axis(energy_values, energy_axes[i], mode=mode, 
                      adjust=adjust, scale="linear")

    n_ticks = 6
    if kalcal_diag_status or kalcal_full_status:
        lbs, ubs = [], []
        if kalcal_diag_status:
            lbs.append(params["kalcal-diag"]["low-bound"])
            ubs.append(params["kalcal-diag"]["up-bound"])
        if kalcal_full_status:
            lbs.append(params["kalcal-full"]["low-bound"])
            ubs.append(params["kalcal-full"]["up-bound"])
        lb = min(lbs)
        ub = max(ubs)
        n_ticks = 2 * (ub - lb)
        place_ticks(axes, bounds=(lb, ub), stype="log", axis="x")
        for ax in axes:
            ax.set_xlim((lb, ub))

        if diag_energy_status or full_energy_status:
            for ax in energy_axes:
                lb, ub = ax.get_ylim()
                place_ticks(ax, bounds=(lb, ub), stype="linear", 
                            axis="y", precision=4)
                
                
    if quartical_status:
        t_ints = params["quartical"]["t-ints"]
        lb = min(t_ints)
        ub = max(t_ints)
        place_ticks(quart_axes, bounds=(lb, ub), 
                    n_ticks=n_ticks, stype="linear")
        if lb < 10:
            lb = 0
        for ax in quart_axes:
            ax.set_xlim((ub, lb))

    fig.subplots_adjust(**subplot_adjust)
    if not diag_energy_status and not full_energy_status:
        place_grid(axes, axis="both")
    else:
        place_grid(axes, axis="both")
    place_markers(params, axes)
    place_legend(params, lines, fig)

    # Show plot
    logging.info(f"Saving figure to `{plot_path}`")
    plt.savefig(plot_path, dpi=plot_options["dpi"])
    params.save()

    if kwargs.get("show", False):
        logging.info("Showing plot")
        plt.show()