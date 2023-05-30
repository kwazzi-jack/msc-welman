import numpy as np
import os
import logging
from tqdm.auto import tqdm

def cm2in(n):
    return n * 0.393701


class DataExistsError(Exception):
    pass


def check_for_data(*paths):
    if len(paths) == 0:
        raise DataExistsError("Empty input path.")
    
    files_exist = [os.path.exists(path) for path in paths]
    if np.any(files_exist):
        total_true = np.sum(files_exist)
        logging.warning(f"The following `{total_true}` files/folders already exist:")
        for path, state in zip(paths, files_exist):
            if state:
                logging.warning(f"Delete? `{path}`")

        choice = input("Do you wish to replace it? (y/n) ")

        if len(choice) > 0 and choice.lower()[0] == "y":
            return 0
            
        else:
            raise DataExistsError("No action is taken.")

    return 0

def progress_bar(prefix=""):
    if len(prefix):
        prefix = prefix + ":"

    pbar = tqdm(
            bar_format=prefix + " {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    return pbar