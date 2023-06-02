import os
from pathlib import Path
from typing import Any
import yaml
import pickle
from copy import copy
from datetime import datetime
from source.other import check_for_data
from contextlib import contextmanager
from dataclasses import dataclass
from ipywidgets import (
    IntText,
    FloatText,
    Text,
    Button,
    HTML,
    AppLayout,
    HBox,
    VBox,
    Layout,
    Dropdown
)
import logging


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
        logging.debug(f"Settings object created: `{name}`")

    def __repr__(self) -> str:
        return str(self.to_dict())
    
    def __str__(self) -> str:
        return str(self.to_dict())

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
        logging.debug(f"Setting added: `{key}`")

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

    def __load(self, btn=None):
        time_stamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        try:
            if btn != None:
                btn.disabled = True
            if self.__immutable:
                path = self.to_yaml(self.__name)
            else:
                path = self.to_yaml(self.__config.value)
                self.__name = self.__config.value
            if btn != None:
                btn.disabled = False
            self.__status.value = f"[{time_stamp}] Loaded: <tt>{path}</tt>"
        except Exception as error:
            if btn != None:
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
            + [HBox([HTML(f"<b>{label}:</b>"), HTML(f"<em>{desc}</em>")]) 
               for label, desc in zip(self.__labels, self.__descriptions)]
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
            pane_widths=[1, 3, 2],
            layout=Layout(width="auto", height="40%"),
            justify_items="left",
        )

        return app

class YamlLoader(yaml.SafeLoader):
    pass

def path_constructor(loader, node):
    value = loader.construct_sequence(node)
    return Path(*value)

YamlLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', path_constructor)

class YamlDict(dict):
    def __init__(self, path, **kwargs):
        logging.debug(f"Creating YamlDict for `{path}`")
        keys = kwargs.keys()
        self.__overwrite = kwargs["overwrite"] if "overwrite" in keys else False
        self.__freeze = False
        self.__path = Path(path)
        super().__init__({})

        if not self.__overwrite:
            try:
                self.__load()
                logging.debug("Load from file")
            except:
                logging.debug("Load empty")
                
        self.__freeze = kwargs["freeze"] if "freeze" in keys else False
        logging.debug(f"Frozen? {self.__freeze}")
    
    @property
    def path(self):
        return self.__path
    
    def freeze(self):
        if not self.__freeze:
            logging.debug("Set to freeze")
            self.__save()
            self.__freeze = True
        
    def unfreeze(self):
        if self.__freeze:
            logging.debug("Set to unfreeze")
            self.__save()
            self.__freeze = False

    def __load(self):
        with open(self.__path, "r") as file:
            new_dict = yaml.load(file, Loader=YamlLoader)
        print(new_dict)
        for key, value in new_dict.items():
            self.__setitem__(key, value)

    def __save(self):
        with open(self.__path, "w") as file:
            file.write(yaml.dump(self.copy()))

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        if not self.__freeze:
            self.__save()
        logging.debug(f"`{self.__path}` added: `{key}`")
        
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


class SmartFolder:
    def __init__(self, path="") -> None:
        if isinstance(path, str):
            path = Path(path)
        self.__path = path
        self.__file_names = []
        self.__files = []
        self.__folder_names = []
        self.__folders = []
        
        if self.__path.exists():
            for item in map(Path, os.listdir(self.__path)):
                full_item = self.__path / item
                if full_item.is_file():
                    name = os.path.splitext(item)[0].replace("-", "_")
                    self.__file_names.append(name)
                    self.__files.append(full_item)
                elif full_item.is_dir():
                    self.__folder_names.append(full_item.name)
                    self.__folders.append(SmartFolder(full_item))
        else:
            self.__path.mkdir(parents=True)

    def __getattr__(self, name):
        name = name.replace("-", "_")

        try:
            if name in self.__file_names:
                idx = self.__file_names.index(name)
                return self.__files[idx]
            elif name in self.__folder_names:
                idx = self.__folder_names.index(name)
                return self.__folders[idx]
        except:
            raise FileNotFoundError(f"The folder/file labelled {name} does not exist.")
            
    def __call__(self) -> Path:
        return self.__path
    
    @property
    def files(self):
        return self.__files
    
    @property
    def folders(self):
        return self.__folders
    
    def refresh(self):
        self = SmartFolder(self.__path)

    def remove(self, name):
        if name in self.__file_names:
            idx = self.__file_names.index(name)
            del self.__file_names[idx]
            del self.__files[idx]

    def exists(self):
        return self.__path.exists() 
    
    def __repr__(self):
        return self._tree_repr()
    
    def _tree_repr(self, level=0, is_last=False):
        tree = ""
        indent = "│   " * level
        
        if level > 0:
            tree += f"{indent}"
            
            if is_last:
                tree += "└── "
            else:
                tree += "├── "
        
        tree += f"{self.__path.name}/\n"
        
        if is_last:
            indent += "    "
        else:
            indent += "│   "
        
        for i, file in enumerate(self.__files):
            if i == len(self.__files) - 1:
                tree += f"{indent}└── {file.name}\n"
            else:
                tree += f"{indent}├── {file.name}\n"
        
        for i, folder in enumerate(self.__folders):
            if i == len(self.__folders) - 1:
                tree += folder._tree_repr(level + 1, is_last=True)
            else:
                tree += folder._tree_repr(level + 1)
        
        return tree
    
def get_parameters(ms_options, gains_options, options, clean=False, force=False):
    pass