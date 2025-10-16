import pandas as pd
from netCDF4 import Dataset
import numpy as np

###################################################################
# REQUIRED USER INPUTS:

# Line 14: Define folder path for lwa_block_summary.19502024.csv
# Line 15: Define folder path for filtered_block_id.19502024.nc
# Line 16: Define folder path for filtered_block_id.19502024.lwa.nc
###################################################################

# File paths
csv_path = "/folder_path/lwa_block_summary.19502024.csv"
nc_in_path = "/folder_path/filtered_block_id.19502024.nc"
nc_out_path = "/folder_path/filtered_block_id.19502024.lwa.nc"

# 1. Read CSV and get cutoff low block_ids

try:
    df = pd.read_csv(csv_path) 
    cutoff_low_ids = set(df.loc[df['classification'] == 'cutoff low', 'block_id'])
except KeyError as e:
    print(f"KeyError: {e}. Columns in CSV: {df.columns.tolist()}")
    raise

# 2. Open input NetCDF and process in chunks for memory efficiency
from collections import defaultdict

with Dataset(nc_in_path, 'r') as src:
    time_dim = src.dimensions['time'].size
    lat_dim = src.dimensions['latitude'].size
    lon_dim = src.dimensions['longitude'].size

    # First pass: find all unique block ids (excluding cutoff low and 0)
    chunk_size = 100  # Adjust as needed for memory
    unique_ids = set()
    for t_start in range(0, time_dim, chunk_size):
        t_end = min(t_start + chunk_size, time_dim)
        object_id_chunk = src.variables['object_id'][t_start:t_end, :, :]
        # Mask out cutoff low blocks
        mask = np.isin(object_id_chunk, list(cutoff_low_ids))
        filtered_chunk = np.where(mask, 0, object_id_chunk)
        unique_ids.update(np.unique(filtered_chunk))
    unique_ids.discard(0)

    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids), start=1)}
    # Print all block id reassignments
    for old_id, new_id in id_map.items():
        if old_id != new_id:
            print(f"{old_id} -> {new_id}")

    # 3. Write to new NetCDF
    with Dataset(nc_out_path, 'w') as dst:
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
        # Copy variables except object_id
        for name, var in src.variables.items():
            if name == 'object_id':
                continue
            out_var = dst.createVariable(name, var.datatype, var.dimensions)
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            out_var[:] = var[:]
        # Create output object_id variable
        out_var = dst.createVariable('object_id', 'i4', ('time', 'latitude', 'longitude'))
        out_var.setncatts({k: src.variables['object_id'].getncattr(k) for k in src.variables['object_id'].ncattrs()})
        # Second pass: process and write in chunks
        for t_start in range(0, time_dim, chunk_size):
            t_end = min(t_start + chunk_size, time_dim)
            object_id_chunk = src.variables['object_id'][t_start:t_end, :, :]
            mask = np.isin(object_id_chunk, list(cutoff_low_ids))
            filtered_chunk = np.where(mask, 0, object_id_chunk)
            remapped_chunk = np.zeros_like(filtered_chunk, dtype=np.int32)
            for old_id, new_id in id_map.items():
                remapped_chunk[filtered_chunk == old_id] = new_id
            out_var[t_start:t_end, :, :] = remapped_chunk
