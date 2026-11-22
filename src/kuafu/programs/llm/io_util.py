import numpy as np
import os
import copy
import json
import yaml
from pathlib import Path
from typing import Iterable


############ IO ################
def load_json(json_file):
    data = None
    with open(json_file, "r") as fin:
        data = json.load(fin)

    return data


def load_txt(file):
    with open(file) as f:
        return f.read().strip()


def write_txt(file, content):
    with open(file, "w") as f:
        f.write(content)


def dump_json(json_data, output_dir):
    with open(output_dir, "w") as fout:
        fout.write(json.dumps(json_data, indent=2))


def load_yaml(yaml_file):
    config = None
    with open(yaml_file, "r") as fin:
        config = yaml.safe_load(fin)

    return config


def save_npz(data, output_dir, compressed=False):
    if compressed:
        np.savez_compressed(output_dir, **data)
    else:
        np.savez(output_dir, **data)


def load_npz(data_file, allow_pickle=True):
    return np.load(data_file, allow_pickle=allow_pickle)


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


def print_config(config, depth=0):
    config_str = ""
    for k, v in config.items():
        if isinstance(v, dict):
            config_str += "{}* {}\n:".format("  " * depth, k)
            config_str += print_config(v, depth + 1)
        else:
            config_str += "{}* {}: {}\n".format("  " * depth, k, v)

    return config_str


def extend_list(list_to_extend, target_len):
    while 2 * len(list_to_extend) < target_len:
        list_to_extend += list_to_extend

    if len(list_to_extend) >= target_len:
        return list_to_extend[0:target_len]
    else:
        return list_to_extend + list_to_extend[0 : target_len - len(list_to_extend)]


def merge_number_dict(mydict, dict_to_merge):
    for k, v in dict_to_merge.items():
        if k not in mydict:
            # Important to use deep copy!
            mydict[k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            merge_number_dict(mydict[k], v)
        else:
            mydict[k] += v


def subtract_number_dict(mydict, dict_to_subtract):
    for k, v in dict_to_subtract.items():
        assert k in mydict, "Can't substract dict!"
        if isinstance(v, dict):
            subtract_number_dict(mydict[k], v)
        else:
            mydict[k] -= v


def subfolders_with_prefix(
    root: str | Path, prefix: str, *, case_sensitive: bool = True
) -> list[Path]:
    root = Path(root)
    dirs: Iterable[Path] = (p for p in root.iterdir() if p.is_dir())
    if case_sensitive:
        return [p for p in dirs if p.name.startswith(prefix)]
    low = prefix.lower()
    return [p for p in dirs if p.name.lower().startswith(low)]


def list_files_by_extension(folder_path, extension) -> list:
    """
    Lists all files in a folder with the given extension.

    :param folder_path: Path to the folder to search in.
    :param extension: File extension to filter by (e.g., '.txt', '.csv').
    :return: List of matching file paths.
    """
    matching_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(extension):
            matching_files.append(os.path.join(folder_path, file))

    return matching_files


def list_sorted_files_by_extension(folder_path, extension):
    """
    Lists all files in a folder with the given extension, sorted by filename.

    :param folder_path: Path to the folder to search in.
    :param extension: File extension to filter by (e.g., '.txt', '.csv').
    :return: List of matching file paths, sorted by filename.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return []

    matching_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(extension)
    ]

    return sorted(
        matching_files, key=lambda x: os.path.basename(x).lower()
    )  # Sorting by filename (case insensitive)
