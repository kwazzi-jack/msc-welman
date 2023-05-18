import numpy as np
import os

def cm2in(n):
    return n * 0.393701


class DataExistsError(Exception):
    pass


def check_data(*paths):    
    files_exist = [os.path.exists(path) for path in paths]
    if np.any(files_exist):
        total_true = np.sum(files_exist)
        print(f"[Warning] The following `{total_true}` files/folders already exist:")
        counter = 0            
        for i, (path, state) in enumerate(zip(paths, files_exist)):
            if state and (counter < 4 or counter > total_true - 5):
                counter += 1
                print(f"----> `{path}`")
                
                if counter == 4:
                    print(" ·\n ·\n ·")
                    
            elif state:
                counter += 1
                
        choice = input("Do you wish to replace it? (y/n) ")
        print("===================================")

        if len(choice) > 0 and choice.lower()[0] == "y":
            return 0
        else:
            raise DataExistsError("No action is taken.")
        
    return 0