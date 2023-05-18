import os
from pathlib import Path
import yaml
from copy import copy
from datetime import datetime
from ipywidgets import (
    Label,
    IntText,
    FloatText,
    BoundedIntText,
    BoundedFloatText,
    Text,
    Button,
    Output,
    HTML,
    AppLayout,
    HBox,
    VBox,
    Layout,
    FileUpload,
)


class Settings:
    __items = dict()
    __labels = []
    __descriptions = []

    def __init__(self, header="", description="", path="") -> None:
        self.__desc = description
        self.__head = header
        self.__dir = path

    def create(self, key, label, description, value):
        self.__labels.append(label)
        self.__descriptions.append(description)

        if isinstance(value, int):
            item = IntText(
                value,
                disabled=False,
                layout=Layout(width="30%"),
            )
            self.__items[key] = item
        elif isinstance(value, float):
            item = FloatText(
                value,
                disabled=False,
                layout=Layout(width="30%"),
            )
            self.__items[key] = item
        elif isinstance(value, str):
            item = Text(
                value,
                disabled=False,
                layout=Layout(width="60%"),
            )
            self.__items[key] = item

    def __setitem__(self, key, item):
        self.create(key, *item)

    def __getitem__(self, key):
        if isinstance(key, list) or isinstance(key, tuple):
            return tuple(self.__items[idx].value for idx in key)
        else:
            return self.__items[key].value

    def from_yaml(self, path):
        with open(path, "r") as file:
            items = yaml.safe_load(file)

        for key, value in items.items():
            self.__items[key].value = value

    def to_yaml(self, path):
        values = self.to_dict()
        with open(path, "w") as file:
            file.write(yaml.dump(values))

    def to_dict(self):
        values = dict()
        for key, item in self.__items.items():
            values[key] = copy(item.value)
        return values

    def __save(self, btn):
        time_stamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        try:
            btn.disabled = True
            self.to_yaml(self.__config.value)
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Saved to: <tt>{self.__dir}</tt>"
            self.__dir = self.__config.value
        except Exception as error:
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Failed to save: {error}"

    def __load(self, btn):
        time_stamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        try:
            btn.disabled = True
            self.from_yaml(self.__config.value)
            btn.disabled = False
            self.__status.value = (
                f"[{time_stamp}] Loaded: <tt>{self.__config.value}</tt>"
            )
            self.__dir = self.__config.value
        except Exception as error:
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Failed to load: {error}"

    def to_widget(self):
        # Heading and description
        h1 = HTML(f"<h1>{self.__head}</h1>")
        h2 = HTML(f"<p>{self.__desc}</p>", layout=Layout(width="65%"))
        header = VBox([h1, h2])

        # Create left sidebar (labels and key)
        left_header = HTML(f"<h2>Index</h2>")

        save_button = Button(description="Save")
        save_button.on_click(self.__save)
        load_button = Button(description="Load")
        load_button.on_click(self.__load)

        self.__status = HTML("")
        self.__config = Text(value=self.__dir)

        left_sidebar = VBox(
            [left_header]
            + [HTML(f"<tt>{key}</tt>") for key in self.__items.keys()]
            + [save_button, load_button]
        )

        # Create right sidebar (descriptions)
        center_bar = VBox(
            [HTML("<h2>Description</h2>")]
            + [HTML(f"<em>{desc}</em>") for desc in self.__descriptions]
            + [self.__config, self.__status]
        )

        # Create center bar (inputs)
        right_sidebar = VBox(
            [HTML("<h2>Input</h2>")] + [item for item in self.__items.values()]
        )

        # Create app layout widget
        app = AppLayout(
            header=header,
            left_sidebar=left_sidebar,
            center=center_bar,
            right_sidebar=right_sidebar,
            pane_widths=[1, 2, 3],
            layout=Layout(width="auto", height="40%"),
            justify_items="left",
        )

        return app


def script_settings(path="config.yml"):
    settings = Settings(
        header="Script Settings",
        description="""Relates to all the system settings relating to the script.
        This covers the entire script from top to bottom and
        so please set carefully. Please correctly set these parameters
        to ensure no over usage of processing, no incorrect paths and
        replicability of data terms.""",
        path=path,
    )

    settings["name"] = (
        "Identifier/Name",
        "Specific name to assign to a simulation run.",
        "meerkat-src-100",
    )

    settings["config-dir"] = (
        "Config file directory",
        "Folder for all generated configuration files.",
        "",
    )

    settings["data-dir"] = (
        "Data directory",
        "Path to directory to hold data files.",
        "",
    )

    settings["plots-dir"] = (
        "Plots directory",
        "Directory to place figures, images and plots.",
        "",
    )

    display(settings.to_widget())
    settings.from_yaml(path)

    return settings

def generate_dir_skeleton(options, remove=False):
    main_dir = Path(options["name"])
    config_dir = main_dir / "config" \
        if len(options["config-dir"]) == 0 else Path(options["config-dir"])
    data_dir = main_dir / "data" \
        if len(options["data-dir"]) == 0 else Path(options["data-dir"])
    plots_dir = main_dir / "plots" \
        if len(options["plots-dir"]) == 0 else Path(options["plots-dir"])
    
    dirs = [
        main_dir,
        config_dir,
        data_dir,
        data_dir / "gains" / "kalcal-diag" / "filter",
        data_dir / "gains" / "kalcal-diag" / "smoother",
        data_dir / "gains" / "kalcal-full" / "filter",
        data_dir / "gains" / "kalcal-full" / "smoother", 
        data_dir / "gains" / "quartical",
        data_dir / "gains" / "true",
        data_dir / "fluxes" / "kalcal-diag" / "filter",
        data_dir / "fluxes" / "kalcal-diag" / "smoother",
        data_dir / "fluxes" / "kalcal-full" / "filter",
        data_dir / "fluxes" / "kalcal-full" / "smoother", 
        data_dir / "fluxes" / "quartical",
        data_dir / "fluxes" / "true",
        data_dir / "fits" / "kalcal-diag" / "filter",
        data_dir / "fits" / "kalcal-diag" / "smoother",
        data_dir / "fits" / "kalcal-full" / "filter",
        data_dir / "fits" / "kalcal-full" / "smoother", 
        data_dir / "fits" / "quartical",
        data_dir / "fits" / "true",
        data_dir / "runs",
        plots_dir,
        plots_dir / "figures",
        plots_dir / "other"
    ]

    for path in dirs:
        os.makedirs(path, exist_ok=True)