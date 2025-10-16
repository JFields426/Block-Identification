import numpy as np
import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

############################################################################
# REQUIRED USER INPUTS:

# Line 16: Define file path for the filtered Block ID file
# Line 18: Define file path for the reference 500hPa geopotential height file
# Line 205: Define the file path for the output .csv file
###########################################################################

####### Load input files #######
nc_block = netCDF4.Dataset('/folder_path/filtered_block_id.19502024.nc', 'r')

nc_hgt = netCDF4.Dataset('/folder_path/ERA5_hgt_500mb.19502024_optimized.nc', 'r')
hgt = nc_hgt.variables['hgt'][:]

####### Align spatial and temporal coordinates #######
block_lat = nc_block.variables['latitude'][:]
block_lon = nc_block.variables['longitude'][:]
hgt_lat = nc_hgt.variables['latitude'][:]
hgt_lon = nc_hgt.variables['longitude'][:]

time_var = nc_block.variables['time']
time_origin = datetime(1900, 1, 1)
times = np.array([time_origin + timedelta(days=float(t)) for t in time_var[:]])
time_resolution_days = (times[1] - times[0]).total_seconds() / (24 * 3600)

# Define weighted areas
dlat = np.radians(np.abs(block_lat[1] - block_lat[0]))
dlon = np.radians(np.abs(block_lon[1] - block_lon[0]))
lat_rad = np.radians(block_lat)
area_weights = np.outer(np.cos(lat_rad), np.ones(len(block_lon))) * dlat * dlon
R = 6371  # Earth radius in km

lat_mask = np.isin(hgt_lat, block_lat)
hgt_lat_indices = np.where(lat_mask)[0]

####### Variables #######
object_id = nc_block.variables['object_id']
n_time = object_id.shape[0]

####### First Pass: Track block presence #######
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

####### Second Pass: Analyze blocks and compute BI #######
print("Second pass: analyzing blocks...")

block_data = {}
BI_values = {}

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

    lat_MZ = block_lat[y_MZ]
    lon_MZ = block_lon[x_MZ] % 360
    lat_idx = hgt_lat_indices[y_MZ]

    # If MZ is at 35.0N, use next latitude above (35.5N)
    if lat_MZ == 35.0:
        target_lat = 35.5
        next_lat_idx = np.argmin(np.abs(hgt_lat - target_lat))
        if np.abs(hgt_lat[next_lat_idx] - target_lat) > 0.01:
            print(f"Block {block}: Unable to find suitable latitude near 35.5N.")
            continue
        MZ = hgt[t_MZ, next_lat_idx, x_MZ]
        hgt_slice = hgt[t_MZ, next_lat_idx, :]
        lat_MZ = hgt_lat[next_lat_idx]
    else:
        MZ = hgt[t_MZ, lat_idx, x_MZ]
        hgt_slice = hgt[t_MZ, lat_idx, :]

    west_lons, east_lons = get_lon_indices_within_range(block_lon, lon_MZ, 90)
    Zu_vals = hgt_slice[west_lons]
    Zd_vals = hgt_slice[east_lons]

    if Zu_vals.size == 0 or Zd_vals.size == 0:
        print(f"Block {block} skipped due to empty longitude selection.")
        continue

    Zu = np.min(Zu_vals)
    Zd = np.min(Zd_vals)

    RC = (((Zu + MZ) / 2) + ((Zd + MZ) / 2)) / 2
    if RC == 0:
        print(f"Block {block}: Skipped due to zero RC.")
        BI = np.nan
    else:
        BI = 100 * ((MZ / RC) - 1)
        if np.isclose(BI, 0, atol=1e-2):
            print(f"BI nearly zero for block {block}: MZ={MZ}, Zu={Zu}, Zd={Zd}, RC={RC}, lat_MZ={lat_MZ}")

    BI_values[block] = BI

    # Center lat/lon at first time step of blocking event
    first_timestep = np.min(times_arr)
    first_indices = np.where(times_arr == first_timestep)[0]

    if len(first_indices) > 0:
        first_y_coords = y_arr[first_indices]
        first_x_coords = x_arr[first_indices]

        center_lat = np.mean(block_lat[first_y_coords])

        lons_rad = np.deg2rad(block_lon[first_x_coords] % 360)
        center_lon_rad = np.arctan2(np.mean(np.sin(lons_rad)), np.mean(np.cos(lons_rad)))
        center_lon = np.rad2deg(center_lon_rad) % 360
    else:
        center_lat = np.nan
        center_lon = np.nan


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
        'center_latitude': center_lat,
        'center_longitude': center_lon,
    }

####### Save to output CSV file #######
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

output_path = '/folder_path/block_summary.19502024.csv'
df.to_csv(output_path, index=False)
print(f"\nCSV file saved to {output_path}")

nc_block.close()
nc_hgt.close()
