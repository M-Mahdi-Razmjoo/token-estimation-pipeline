import sys
import pandas
import numpy
import sklearn
import pyarrow
import tqdm
import regex

def get_environment_info():
    info_str = "environment & library versions\n"
    info_str += "-"*60

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info_str += f"{'Python Version:':<20} {py_version}\n"

    libraries = {
        "scikit-learn": sklearn,
        "pandas": pandas,
        "numpy": numpy,
        "pyarrow": pyarrow,
        "tqdm": tqdm,
        "regex": regex
    }
    
    for name, lib in libraries.items():
        info_str += f"{name.capitalize() + ' Version:':<20} {lib.__version__}\n"
            
    info_str += "-"*60
    return info_str