import sys
import platform
from importlib import metadata

def get_environment_info():
    info_str = "Environment, Python & Library Versions\n"
    info_str +="-"*60+"\n"
    info_str += f"{'OS Platform:':<25} {platform.platform()}\n"
    py_version = sys.version.replace('\n', ' ')
    py_arch = platform.architecture()[0]
    info_str += f"{'Python Version:':<25} {py_version}\n"
    info_str += f"{'Python Architecture:':<25} {py_arch}\n"
    info_str += "Installed Packages & Versions in Environment:\n"
    info_str +="-"*60+"\n"
    installed_packages = sorted([f"{dist.metadata['Name']} ({dist.version})" for dist in metadata.distributions()],key=str.lower)
    col_width = max(len(p) for p in installed_packages) + 4
    num_cols = max(1, 100 // col_width)
    rows = []
    for i in range(0, len(installed_packages), num_cols):
        row_items = installed_packages[i:i+num_cols]
        padded_items = [item.ljust(col_width) for item in row_items]
        rows.append("".join(padded_items))
    info_str +="\n".join(rows)
    info_str +="-"*60+"\n\n"
    return info_str