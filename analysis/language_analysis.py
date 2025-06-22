import os
import glob
import torch
import re
from typing import List, Optional, Union, Tuple

def load_latest_interaction(
    logs_root: str,
    seed_folder: str,
    split: str = "validation",
    prefix: str = "interaction_gpu0",
    map_to_cpu: bool = True
) -> object:
    base_pattern = os.path.join(
        logs_root,
        seed_folder,
        "interactions",
        split,
        "epoch_*",
        f"{prefix}*"
    )
    all_files: List[str] = glob.glob(base_pattern)
    if not all_files:
        raise FileNotFoundError(
            f"No files matching the pattern:\n  {base_pattern}\nwere found."
        )
    
    def epoch_num_from_path(fp: str) -> int:
        parent_dir = os.path.basename(os.path.dirname(fp))
        if not parent_dir.startswith("epoch_"):
            raise ValueError(f"Unexpected folder name (not epoch_<N>): {parent_dir!r}")
        try:
            return int(parent_dir.split("_", 1)[1])
        except ValueError:
            raise ValueError(f"Cannot parse epoch number from: {parent_dir!r}")

    try:
        all_files_sorted = sorted(all_files, key=epoch_num_from_path)
    except ValueError as e:
        raise ValueError(f"Error parsing epoch folders: {e}")

    last_file = all_files_sorted[-1]
    print(last_file)

    if map_to_cpu:
        return torch.load(last_file, map_location="cpu")
    else:
        return torch.load(last_file)

def load_all_interactions(
    logs_root: str,
    seed_folder: str,
    split: str = "validation",
    prefix: str = "interaction_gpu0",
    map_to_cpu: bool = True
) -> List[object]:
    """
    Loads all interaction files from every epoch.
    """
    base_pattern = os.path.join(
        logs_root,
        seed_folder,
        "interactions",
        split,
        "epoch_*",
        f"{prefix}*"
    )
    all_files: List[str] = glob.glob(base_pattern)

    if not all_files:
        raise FileNotFoundError(
            f"No files matching the pattern:\n  {base_pattern}\nwere found."
        )

    def epoch_num_from_path(fp: str) -> int:
        """Extracts the epoch number from a file path."""
        parent_dir = os.path.basename(os.path.dirname(fp))
        if not parent_dir.startswith("epoch_"):
            raise ValueError(f"Unexpected folder name (not epoch_<N>): {parent_dir!r}")
        try:
            return int(parent_dir.split("_", 1)[1])
        except ValueError:
            raise ValueError(f"Cannot parse epoch number from: {parent_dir!r}")

    try:
        all_files_sorted = sorted(all_files, key=epoch_num_from_path)
    except ValueError as e:
        raise ValueError(f"Error parsing epoch folders: {e}")

    loaded_interactions = []
    for file_path in all_files_sorted:
        print(f"Loading file: {file_path}")
        if map_to_cpu:
            loaded_interactions.append(torch.load(file_path, map_location="cpu"))
        else:
            loaded_interactions.append(torch.load(file_path))

    return loaded_interactions


