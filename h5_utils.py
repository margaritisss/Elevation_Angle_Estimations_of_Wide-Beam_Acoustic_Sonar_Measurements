import os
import h5py
import numpy as np
from types import SimpleNamespace


def save(filepath, group_path, data, mode='a', compress=True, compression_opts=4):
    """
    Save `data` under the nested group `group_path` in an HDF5 file.
    Creates any intermediate groups if they don't exist, and overwrites
    existing datasets with the same names.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file. Parent directories will be created as needed.
    group_path : str
        Slash-delimited HDF5 group path, e.g. 'PARENT/CHILD'.
    data : object, dict, or list of dicts
        - If object with __dict__: dumps each attribute as a dataset.
        - If dict: each key/value pair becomes a dataset.
        - If list of dicts: creates subgroups '0001', '0002', ... under group_path, each dict saved inside.
    mode : {'w','a'}, optional
        File mode, 'w' to overwrite or 'a' to append/modify (default 'a').
    compress : bool, optional
        Whether to gzip-compress each dataset (default True).
    compression_opts : int, optional
        Gzip compression level, 1–9 (default 4).
    """
    # ensure output directory exists
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(filepath, mode) as f:
        # create or retrieve nested group
        grp = f
        for part in group_path.split('/'):
            grp = grp.require_group(part)

        def _write(g, name, arr):
            # convert to numpy array
            arr = np.asarray(arr)
            # handle object arrays (e.g. strings)
            if arr.dtype == object:
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    dt = h5py.string_dtype(encoding='utf-8')
                    if name in g:
                        del g[name]
                    g.create_dataset(name, data=np.array(arr, dtype=object), dtype=dt)
                    return

            # overwrite existing
            if name in g:
                del g[name]
            if compress:
                g.create_dataset(name, data=arr, compression='gzip', compression_opts=compression_opts)
            else:
                g.create_dataset(name, data=arr)

        # dispatch based on data type
        if hasattr(data, '__dict__'):
            for field, val in vars(data).items():
                _write(grp, field, val)

        elif isinstance(data, dict):
            for field, val in data.items():
                _write(grp, field, val)

        elif isinstance(data, list):
            for idx, item in enumerate(data, 1):
                subgrp = grp.require_group(f"{idx:04d}")
                if not isinstance(item, dict):
                    raise ValueError("List items must be dicts to be saved as subgroups")
                for field, val in item.items():
                    _write(subgrp, field, val)

        else:
            raise ValueError("`data` must be an object with __dict__, a dict, or a list of dicts.")

    print(f"Saved to {filepath} -> {group_path}")


def load(filename, group_name=None, return_all_groups=False, update_globals=False):
    """
    Load data from an HDF5 file in the HDF5_DATA folder.
    
    Parameters
    ----------
    filename : str
        Name of the HDF5 file (e.g., '1_RAW_DATA.h5', '3_VALID_DATA.h5')
    group_name : str, optional
        Specific group to load. If None and return_all_groups=False, returns the first group.
    return_all_groups : bool, optional
        If True, returns a dictionary with all groups as SimpleNamespace objects.
    update_globals : bool, optional
        If True, updates the global namespace with the loaded groups.
        
    Returns
    -------
    SimpleNamespace or dict
        - If group_name is specified: SimpleNamespace with the group's datasets
        - If return_all_groups=True: dict mapping group names to SimpleNamespace objects
        - If update_globals=True: None (updates globals directly)
        
    Examples
    --------
    # Load a specific group
    RAW_DVL = load('1_RAW_DATA.h5', 'RAW_DVL')
    
    # Load all groups as a dictionary
    all_data = load('3_VALID_DATA.h5', return_all_groups=True)
    DVL_VALID = all_data['DVL_VALID']
    
    # Update global namespace with all groups
    load('3_VALID_DATA.h5', update_globals=True)
    # Now DVL_VALID, ECHO_VALID, etc. are available as global variables
    """
    # Construct the full filepath
    filepath = os.path.join('HDF5_DATA', filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        if group_name:
            # Load a specific group
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found in {filename}")
            grp = f[group_name]
            datasets = {ds: grp[ds][:] for ds in grp.keys() if isinstance(grp[ds], h5py.Dataset)}
            result = SimpleNamespace(**datasets)
            print(f"Loaded {group_name} from {filename}")
            return result
            
        elif return_all_groups:
            # Load all groups as a dictionary
            result = {}
            for grp_name in f.keys():
                grp = f[grp_name]
                datasets = {ds: grp[ds][:] for ds in grp.keys() if isinstance(grp[ds], h5py.Dataset)}
                result[grp_name] = SimpleNamespace(**datasets)
            print(f"Loaded all groups from {filename}")
            return result
            
        elif update_globals:
            # Update global namespace with all groups
            for grp_name in f.keys():
                grp = f[grp_name]
                datasets = {ds: grp[ds][:] for ds in grp.keys() if isinstance(grp[ds], h5py.Dataset)}
                globals()[grp_name] = SimpleNamespace(**datasets)
            print(f"Updated globals with all groups from {filename}")
            return None
            
        else:
            # Default: return the first group
            if len(f.keys()) == 0:
                raise ValueError(f"No groups found in {filename}")
            first_group = list(f.keys())[0]
            grp = f[first_group]
            datasets = {ds: grp[ds][:] for ds in grp.keys() if isinstance(grp[ds], h5py.Dataset)}
            result = SimpleNamespace(**datasets)
            print(f"Loaded {first_group} from {filename}")
            return result

