import os
from pathlib import Path
import yaml
from copy import copy
from datetime import datetime
from source.other import check_for_data
from contextlib import contextmanager
from ipywidgets import (
    IntText,
    FloatText,
    Text,
    Button,
    HTML,
    AppLayout,
    VBox,
    Layout,
    Dropdown
)


class Settings:
    def __init__(self, name="", header="", description="", 
                 directory="", immutable_path=True) -> None:
        self.__desc = description
        self.__head = header
        self.__immutable = immutable_path
        self.__dir = directory
        self.__name = name
        self.__items = dict()
        self.__labels = []
        self.__descriptions = []

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
        elif isinstance(value, list):
            item = Dropdown(
                options=value,
                value=value[0],
                disabled=False
            )
            self.__items[key] = item

    def __setitem__(self, key, item):
        self.create(key, *item)

    def __getitem__(self, key):
        if isinstance(key, list) or isinstance(key, tuple):
            return tuple(self.__items[idx].value for idx in key)
        else:
            return self.__items[key].value
        
    @property
    def path(self):
        if self.__immutable:
            return str(Path(self.__dir) / self.__name)
        else:
            return str(Path(self.__name))

    def from_yaml(self, path):
        if self.__immutable:
            filename = Path(self.__dir) / path
        else:
            filename = Path(path)  
        with open(filename, "r") as file:
            items = yaml.safe_load(file)

        keys = self.__items.keys()
        for key, value in items.items():
            if key in keys:
                self.__items[key].value = value

        return filename

    def to_yaml(self, path):
        if self.__immutable:
            filename = Path(self.__dir) / path
        else:
            filename = Path(path)  
        values = self.to_dict()
        with open(filename, "w") as file:
            file.write(yaml.dump(values))

        return filename

    def to_dict(self):
        values = dict()
        for key, item in self.__items.items():
            values[key] = copy(item.value)
        return values

    def __save(self, btn):
        time_stamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        try:
            btn.disabled = True
            if self.__immutable:
                path = self.to_yaml(self.__name)
            else:
                path = self.to_yaml(self.__config.value)
                self.__name = self.__config.value
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Saved to: <tt>{path}</tt>"
        except Exception as error:
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Failed to save: {error}"

    def __load(self, btn):
        time_stamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        try:
            btn.disabled = True
            if self.__immutable:
                path = self.to_yaml(self.__name)
            else:
                path = self.to_yaml(self.__config.value)
                self.__name = self.__config.value
            btn.disabled = False
            self.__status.value = f"[{time_stamp}] Loaded: <tt>{path}</tt>"
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
        self.__config = Text(value=str(Path(self.__dir) / self.__name), 
                             disabled=self.__immutable)

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
            pane_widths=[1, 3, 3],
            layout=Layout(width="auto", height="40%"),
            justify_items="left",
        )

        return app


class YamlDict(dict):
    def __init__(self, path, **kwargs):
        keys = kwargs.keys()
        self.__overwrite = kwargs["overwrite"] if "overwrite" in keys else False
        self.__freeze = kwargs["freeze"] if "freeze" in keys else False
        self.__path = Path(path)
        super().__init__({})
        if not self.__overwrite:
            try:
                self.__load()
            except:
                pass
    
    @property
    def path(self):
        return self.__path
    
    def freeze(self):
        if not self.__freeze:
            self.__save()
            self.__freeze = True
        
    def unfreeze(self):
        if self.__freeze:
            self.__save()
            self.__freeze = False

    def __load(self):
        with open(self.__path, "r") as file:
            new_dict = yaml.safe_load(file)

        for key, value in new_dict.items():
            self.__setitem__(key, value)

    def __save(self):
        with open(self.__path, "w") as file:
            file.write(yaml.dump(self.copy()))

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        if not self.__freeze:
            self.__save()
        
    def __repr__(self) -> str:
        return super().__repr__()
    
    def __str__(self) -> str:
        return super().__str__()

@contextmanager
def refreeze(x):
    x.unfreeze()
    try:
        yield x
    finally:
        x.freeze() 

def get_parameters(ms_options, gains_options, options, clean=False, force=False):
    pass