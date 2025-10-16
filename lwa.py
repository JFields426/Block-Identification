import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from numpy import cos, sin, pi, arcsin
from tqdm import tqdm
import os

#########################################################
# REQUIRED USER INPUTS:

# Line 24: Define folder path for input files
# Line 40: Define folder path for output file 
#########################################################

# === Manual Date Range Selection ===
start_date = '1950-01-01'
end_date = '2024-12-31'

# === Constants ===
a = 6.371e6  # Earth's radius [m]

# === Load Z500 Data ===
ds = xr.open_dataset("/folder_path/ERA5_hgt_500mb.19502024_optimized.nc")
z500 = ds['hgt'].sel(time=slice(start_date, end_date))
lat = z500.latitude.values
lon = z500.longitude.values
time = z500.time.values

# === Pre-compute lat-dependent metrics ===
phi = np.deg2rad(lat)
cosphi = np.cos(phi)
dphi = np.abs(np.gradient(phi))

# === Zonal mean and anomaly ===
z_zonal_mean = z500.mean(dim='longitude')
z_prime = z500 - z_zonal_mean

# === Output arrays ===
output_path = "/folder_path/lwa.nc"
# Pre-allocate output arrays
lwa_all = np.full((len(time), len(lat), len(lon)), np.nan, dtype=np.float32)
lwa_anti_all = np.full((len(time), len(lat), len(lon)), np.nan, dtype=np.float32)
lwa_cyclo_all = np.full((len(time), len(lat), len(lon)), np.nan, dtype=np.float32)

# === Compute LWA ===
for ti in tqdm(range(len(time)), desc="Computing LWA"):
    lwa_t = np.full((len(lat), len(lon)), np.nan)
    lwa_anti_t = np.full((len(lat), len(lon)), np.nan)
    lwa_cyclo_t = np.full((len(lat), len(lon)), np.nan)
    for li in range(len(lon)):
        # Column of anomalies and full values
        z_col = z_prime.isel(time=ti, longitude=li).values
        z_full = z500.isel(time=ti, longitude=li).values
        # Sort latitude by geopotential height values
        sort_idx = np.argsort(z_full)
        z_sorted = z_full[sort_idx]
        phi_sorted = phi[sort_idx]
        cosphi_sorted = cosphi[sort_idx]
        dphi_sorted = dphi[sort_idx]
        # Compute cumulative area fractions
        area_weights = cosphi_sorted * dphi_sorted
        cumulative_area = np.cumsum(area_weights)
        total_area = np.sum(area_weights)
        area_fraction = cumulative_area / total_area
        # Equivalent latitude
        phi_e = np.arcsin(1 - 2 * np.clip(area_fraction, 0, 1))
        # Interpolate z' back to regular phi grid
        z_prime_sorted = z_col[sort_idx]
        z_interp = interp1d(phi_sorted, z_prime_sorted, kind='linear',
                            bounds_error=False, fill_value=0.0)
        z_regrid = z_interp(phi)
        z_regrid_weighted = z_regrid * cosphi * dphi
        #Compute for all points north of 35N
        for lat_idx in range(len(lat)):
            if lat[lat_idx] < 35:
                continue
            phi_ei = phi[lat_idx]
            cosphi_ei = np.maximum(np.cos(phi_ei), 1e-1)
            # Anticyclonic part (z' ≥ 0, south of equivalent latitude)
            anti_mask = (phi <= phi_ei) & (z_regrid >= 0)
            lwa_anti_val = (a / cosphi_ei) * np.sum(z_regrid_weighted[anti_mask])
            # Cyclonic part (z' ≤ 0, north of equivalent latitude)
            cyclo_mask = (phi >= phi_ei) & (z_regrid <= 0)
            lwa_cyclo_val = -(a / cosphi_ei) * np.sum(z_regrid_weighted[cyclo_mask])
            # Store results
            lwa_t[lat_idx, li] = lwa_anti_val + lwa_cyclo_val
            lwa_anti_t[lat_idx, li] = lwa_anti_val
            lwa_cyclo_t[lat_idx, li] = lwa_cyclo_val
    lwa_all[ti, :, :] = lwa_t
    lwa_anti_all[ti, :, :] = lwa_anti_t
    lwa_cyclo_all[ti, :, :] = lwa_cyclo_t

lwa_xr = xr.DataArray(lwa_all, coords={'time': time, 'latitude': lat, 'longitude': lon},
                      dims=('time', 'latitude', 'longitude'), name='lwa')
lwa_anti_xr = xr.DataArray(lwa_anti_all, coords={'time': time, 'latitude': lat, 'longitude': lon},
                          dims=('time', 'latitude', 'longitude'), name='lwa_anticyclonic')
lwa_cyclo_xr = xr.DataArray(lwa_cyclo_all, coords={'time': time, 'latitude': lat, 'longitude': lon},
                            dims=('time', 'latitude', 'longitude'), name='lwa_cyclonic')
ds_out = xr.Dataset({
    'lwa': lwa_xr,
    'lwa_anticyclonic': lwa_anti_xr,
    'lwa_cyclonic': lwa_cyclo_xr
})
ds_out.to_netcdf(output_path, mode='w')
