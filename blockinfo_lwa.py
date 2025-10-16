import numpy as np
import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import xarray as xr
from tqdm import tqdm

#################################################################
# REQUIRED USER INPUTS:

# Line 20: Define folder path for filtered_block_id.19502024.nc
# Line 25: Define folder path for ERA5_500hgt.19502024_optimized.nc
# Line 30: Define folder path for lwa.nc
# Line 265: Define folder path for output file
#################################################################

# === Load Block ID File ===
nc_block = netCDF4.Dataset(
    '/folder_path/filtered_block_id.19502024.nc', 'r'
)

# === Load hgt File ===
nc_hgt = netCDF4.Dataset(
    '/folder_path/ERA5_500hgt.19502024_optimized.nc', 'r'
)
hgt = nc_hgt.variables['hgt'][:]

# === Load LWA data ===
lwa_ds = xr.open_dataset("/folder_path/lwa.nc")
lwa = lwa_ds['lwa']
lwa_anti = lwa_ds['lwa_anticyclonic']
lwa_cyclo = lwa_ds['lwa_cyclonic']

# === Coordinates and Time ===
block_lat = nc_block.variables['latitude'][:]
block_lon = nc_block.variables['longitude'][:]
hgt_lat = nc_hgt.variables['latitude'][:]
hgt_lon = nc_hgt.variables['longitude'][:]

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

# === Second Pass: Analyze blocks and compute BI and LWA ===

print("Second pass: analyzing blocks...")

block_data = {}
BI_values = {}
lat_MZ_dict = {}
lon_MZ_dict = {}
LWA_data = {}

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

for block, points in tqdm(block_points.items(), desc="Analyzing blocks", total=len(block_points)):
    times_list, y_list, x_list = zip(*points)
    times_arr = np.array(times_list)
    y_arr = np.array(y_list)
    x_arr = np.array(x_list)

    # === BI CALCULATION ===
    hgt_lat_indices = [np.abs(hgt_lat - lat).argmin() for lat in block_lat]
    hgt_vals = hgt[times_arr, [hgt_lat_indices[y] for y in y_arr], x_arr]
    max_index = np.nanargmax(hgt_vals)

    t_MZ = times_arr[max_index]
    y_MZ = y_arr[max_index]
    x_MZ = x_arr[max_index]

    lat_MZ = block_lat[y_MZ]
    lon_MZ = block_lon[x_MZ] % 360
    lat_MZ_dict[block] = lat_MZ
    lon_MZ_dict[block] = lon_MZ

    west_lons, east_lons = get_lon_indices_within_range(block_lon, lon_MZ, 90)
    lat_MZ_idx = np.abs(hgt_lat - lat_MZ).argmin()
    MZ = hgt[t_MZ, lat_MZ_idx, x_MZ]

    if lat_MZ == 35.0:
        lat_below_idx = lat_MZ_idx - 1 if lat_MZ_idx > 0 else None
        if lat_below_idx is not None:
            hgt_slice = hgt[t_MZ, lat_below_idx, :]
        else:
            BI_values[block] = np.nan
            continue
    else:
        hgt_slice = hgt[t_MZ, lat_MZ_idx, :]

    Zu_vals = hgt_slice[west_lons]
    Zd_vals = hgt_slice[east_lons]

    if Zu_vals.size == 0 or Zd_vals.size == 0:
        BI_values[block] = np.nan
    else:
        Zu = np.min(Zu_vals)
        Zd = np.min(Zd_vals)
        RC = (((Zu + MZ) / 2) + ((Zd + MZ) / 2)) / 2
        BI_values[block] = 100 * ((MZ / RC) - 1)

    # === Block Core Info ===
    area_times = block_area[block]
    min_area_km2, min_time = min(area_times, key=lambda x: x[0])
    max_area_km2, max_time = max(area_times, key=lambda x: x[0])

    first_t = min(times_list)
    first_indices = [i for i, t in enumerate(times_list) if t == first_t]
    first_ys = [y_list[i] for i in first_indices]
    first_xs = [x_list[i] for i in first_indices]
    center_latitude = np.mean(block_lat[list(first_ys)])
    lons = block_lon[list(first_xs)] % 360
    lons_rad = np.deg2rad(lons)
    mean_sin = np.mean(np.sin(lons_rad))
    mean_cos = np.mean(np.cos(lons_rad))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    center_longitude = np.rad2deg(mean_angle) % 360

    block_data[block] = {
        'start_time': block_times[block][0],
        'end_time': block_times[block][1],
        'duration_steps': len(set(times_list)),
        'duration_days': len(set(times_list)) * time_resolution_days,
        'min_spatial_area_km2': min_area_km2,
        'min_size_time': min_time,
        'max_spatial_area_km2': max_area_km2,
        'max_size_time': max_time,
        'center_latitude': center_latitude,
        'center_longitude': center_longitude,
    }

    # === LWA CALCULATION (using date_MZ) ===
    # Find t_MZ for this block (time index of max hgt)
    # Already computed above as t_MZ
    mask_t_MZ = times_arr == t_MZ
    y_MZ_arr = y_arr[mask_t_MZ]
    x_MZ_arr = x_arr[mask_t_MZ]

    lwa_vals = []
    lwa_anti_vals = []
    lwa_cyclo_vals = []
    for yi, xi in zip(y_MZ_arr, x_MZ_arr):
        lat_val = block_lat[yi]
        lon_val = block_lon[xi]
        lat_idx = np.abs(lwa.latitude.values - lat_val).argmin()
        lon_idx = np.abs(lwa.longitude.values - lon_val).argmin()
        if t_MZ < lwa.sizes['time']:
            lwa_vals.append(lwa.isel(time=t_MZ, latitude=lat_idx, longitude=lon_idx).item())
            lwa_anti_vals.append(lwa_anti.isel(time=t_MZ, latitude=lat_idx, longitude=lon_idx).item())
            lwa_cyclo_vals.append(lwa_cyclo.isel(time=t_MZ, latitude=lat_idx, longitude=lon_idx).item())

    avg_lwa = np.nanmean(lwa_vals) if lwa_vals else np.nan
    avg_anti = np.nanmean(lwa_anti_vals) if lwa_anti_vals else np.nan
    avg_cyclo = np.nanmean(lwa_cyclo_vals) if lwa_cyclo_vals else np.nan

    if avg_anti > 10 * avg_cyclo:
        classification = "ridge block"
    elif 10 * avg_cyclo > avg_anti > 0.5 * avg_cyclo:
        classification = "dipole block"
    elif avg_anti < 0.5 * avg_cyclo:
        classification = "cutoff low"
    else:
        classification = "unclassified"

    LWA_data[block] = {
        "max_lwa": avg_lwa,
        "time_of_max_lwa": times[t_MZ],
        "avg_LWA_anticyclonic": avg_anti,
        "avg_LWA_cyclonic": avg_cyclo,
        "classification": classification,
    }

# === Merge into DataFrame ===
df = pd.DataFrame.from_dict(block_data, orient='index')
df.index.name = 'block_id'
df = df.reset_index()

# Add date of MZ (maximum hgt for block)
def get_date_of_MZ(block_id):
    points = block_points[block_id]
    times_list, y_list, x_list = zip(*points)
    hgt_lat_indices = [np.abs(hgt_lat - lat).argmin() for lat in block_lat]
    hgt_vals = hgt[list(times_list), [hgt_lat_indices[y] for y in y_list], x_list]
    max_index = np.nanargmax(hgt_vals)
    t_MZ = times_list[max_index]
    return times[t_MZ].date() if hasattr(times[t_MZ], 'date') else times[t_MZ]

df['date_MZ'] = df['block_id'].map(get_date_of_MZ)
df['blocking_index'] = df['block_id'].map(BI_values)
df['lat_MZ'] = df['block_id'].map(lat_MZ_dict)
df['lon_MZ'] = df['block_id'].map(lon_MZ_dict)

df['max_lwa'] = df['block_id'].map(lambda b: LWA_data.get(b, {}).get("max_lwa", np.nan))
df['time_of_max_lwa'] = df['block_id'].map(lambda b: LWA_data.get(b, {}).get("time_of_max_lwa", pd.NaT))
df['avg_LWA_anticyclonic'] = df['block_id'].map(lambda b: LWA_data.get(b, {}).get("avg_LWA_anticyclonic", np.nan))
df['avg_LWA_cyclonic'] = df['block_id'].map(lambda b: LWA_data.get(b, {}).get("avg_LWA_cyclonic", np.nan))
df['classification'] = df['block_id'].map(lambda b: LWA_data.get(b, {}).get("classification", "unclassified"))

# === Save Final CSV ===
cols = list(df.columns)
if 'date_MZ' in cols and 'lat_MZ' in cols:
    lat_idx = cols.index('lat_MZ')
    cols.remove('date_MZ')
    cols.insert(lat_idx, 'date_MZ')
    df = df[cols]

output_path = '/folder_path/lwa_block_summary.19502024.csv'
df.to_csv(output_path, index=False)
print(f"\nCSV file saved to {output_path}")

# === Close Files ===
nc_block.close()
nc_hgt.close()
lwa_ds.close()
