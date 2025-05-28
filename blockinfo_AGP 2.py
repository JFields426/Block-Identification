import numpy as np
import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# === Load Block ID File ===
nc_block = netCDF4.Dataset('/share/data1/Students/jfields/TempestExtremes/BlockID/block_id.filtered.19502022.nc', 'r')

# === Load hgt File ===
nc_hgt = netCDF4.Dataset('/share/data1/Students/jfields/TempestExtremes/ERA5_hgt_500mb.19502022_optimized.nc', 'r')
hgt = nc_hgt.variables['hgt'][:]  # shape (time, lat, lon)

# === Coordinates and Time ===
block_lat = nc_block.variables['latitude'][:]  # (111,) — blocking dataset
block_lon = nc_block.variables['longitude'][:]  # (720,)
hgt_lat = nc_hgt.variables['latitude'][:]       # (361,)
hgt_lon = nc_hgt.variables['longitude'][:]      # (720,)

time_var = nc_block.variables['time']
time_origin = datetime(1900, 1, 1)
times = np.array([time_origin + timedelta(days=float(t)) for t in time_var[:]])
time_resolution_days = (times[1] - times[0]).total_seconds() / (24 * 3600)

# === Area Weights ===
dlat = np.radians(np.abs(block_lat[1] - block_lat[0]))
dlon = np.radians(np.abs(block_lon[1] - block_lon[0]))
lat_rad = np.radians(block_lat)
area_weights = np.outer(np.cos(lat_rad), np.ones(len(block_lon))) * dlat * dlon
R = 6371  # Earth radius in km

# === Match blocking latitudes in hgt file ===
lat_mask = np.isin(hgt_lat, block_lat)
hgt_lat_indices = np.where(lat_mask)[0]  # indices to subset hgt to match block latitudes

# === Variables ===
object_id = nc_block.variables['object_id']
n_time = object_id.shape[0]

# === First Pass: Track block presence ===
block_points = defaultdict(list)
block_times = {}
block_area = defaultdict(list)

print("First pass: collecting block locations...")
for t in range(n_time):
    if t % 100 == 0:
        print(f"Processing timestep {t}/{n_time}")

    tag_slice = object_id[t, :, :]
    valid = ~np.isnan(tag_slice)
    unique_blocks = np.unique(tag_slice[valid])

    for block in unique_blocks:
        if block == 0:
            continue

        indices = np.where(tag_slice == block)
        for y, x in zip(*indices):
            block_points[int(block)].append((t, y, x))

        if block not in block_times:
            block_times[block] = [times[t], times[t]]
        else:
            block_times[block][1] = times[t]

        area_km2 = np.sum(area_weights[indices]) * R**2
        block_area[block].append((area_km2, times[t]))

# === Second Pass: Analyze blocks and compute BI ===
print("Second pass: analyzing blocks...")

block_data = {}
BI_values = {}

# Wraparound-safe longitude mask function
def get_lon_indices_within_range(lon_vals, center_lon, offset):
    lon_vals = lon_vals % 360
    center_lon = center_lon % 360
    min_lon = (center_lon - offset) % 360
    max_lon = (center_lon + offset) % 360

    if min_lon < center_lon:
        west_mask = (lon_vals >= min_lon) & (lon_vals < center_lon)
    else:
        west_mask = (lon_vals >= min_lon) | (lon_vals < center_lon)

    if center_lon < max_lon:
        east_mask = (lon_vals > center_lon) & (lon_vals <= max_lon)
    else:
        east_mask = (lon_vals > center_lon) | (lon_vals <= max_lon)

    return np.where(west_mask)[0], np.where(east_mask)[0]

for block, points in block_points.items():
    times_list, y_list, x_list = zip(*points)

    times_arr = np.array(times_list)
    y_arr = np.array(y_list)
    x_arr = np.array(x_list)

    # Select the hgt values within the block
    hgt_vals = hgt[times_arr, hgt_lat_indices[y_arr], x_arr]
    max_index = np.nanargmax(hgt_vals)

    t_MZ = times_arr[max_index]
    y_MZ = y_arr[max_index]
    x_MZ = x_arr[max_index]

    MZ = hgt[t_MZ, hgt_lat_indices[y_MZ], x_MZ]
    lat_MZ = block_lat[y_MZ]
    lon_MZ = block_lon[x_MZ] % 360

    # === Find longitude ranges for Zu and Zd ===
    west_lons, east_lons = get_lon_indices_within_range(block_lon, lon_MZ, 90)

    hgt_slice = hgt[t_MZ, hgt_lat_indices[y_MZ], :]  # lat-sliced
    Zu_vals = hgt_slice[west_lons]
    Zd_vals = hgt_slice[east_lons]

    # === Compute BI only if valid
    if Zu_vals.size == 0 or Zd_vals.size == 0:
        print(f"Block {block} skipped due to empty longitude selection.")
        continue

    Zu = np.min(Zu_vals)
    Zd = np.min(Zd_vals)

    RC = (((Zu + MZ) / 2) + ((Zd + MZ) / 2)) / 2
    BI = 100 * ((MZ / RC) - 1)

    if lat_MZ == 35.0:
        print(f"Skipping BI for block {block} due to max height at southern edge (lat_MZ = 35.0)")
        BI = np.nan
    elif BI == 0:
        print(f"Zero BI detected for block {block}: MZ={MZ}, Zu={Zu}, Zd={Zd}, RC={RC}, lat_MZ={lat_MZ}")


    BI_values[block] = BI

    # === Other Block Properties ===
    area_times = block_area[block]
    min_area_km2, min_time = min(area_times, key=lambda x: x[0])
    max_area_km2, max_time = max(area_times, key=lambda x: x[0])

    block_data[block] = {
        'start_time': block_times[block][0],
        'end_time': block_times[block][1],
        'duration_steps': len(set(times_list)),
        'duration_days': len(set(times_list)) * time_resolution_days,
        'min_spatial_area_km2': min_area_km2,
        'min_size_time': min_time,
        'max_spatial_area_km2': max_area_km2,
        'max_size_time': max_time,
        'center_latitude': np.mean(block_lat[list(y_list)]),
        'center_longitude': np.mean(block_lon[list(x_list)]),
    }

# === Save to CSV ===
df = pd.DataFrame.from_dict(block_data, orient='index')
df.index.name = 'block_id'
df = df.reset_index()

# Add BI column
df['blocking_index'] = df['block_id'].map(BI_values)

# Reorder columns
df = df[
    ['block_id', 'start_time', 'end_time', 'duration_days', 'duration_steps',
     'min_spatial_area_km2', 'min_size_time', 'max_spatial_area_km2', 'max_size_time',
     'center_latitude', 'center_longitude',
     'blocking_index']
]

output_path = '/share/data1/Students/jfields/TempestExtremes/BlockID/AGP_block_summary.19502022_BI.csv'
df.to_csv(output_path, index=False)
print(f"\nCSV file saved to {output_path}")

# === Close Files ===
nc_block.close()
nc_hgt.close()
