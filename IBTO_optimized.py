import xarray as xr
import numpy as np
import glob
import os

# Define the latitude ranges
lat0_range = np.arange(35, 90.5, 0.5)
latS_range = np.where(lat0_range > 70, lat0_range - (90 - lat0_range), lat0_range - 20)
latN_range = np.where(lat0_range > 70, lat0_range + (90 - lat0_range), lat0_range + 20)
lat15S_range = np.where(lat0_range > 70, lat0_range - (90 - lat0_range) * 0.75, lat0_range - 15)
lat30S_range = np.where(lat0_range > 70, lat0_range - (90 - lat0_range) * 1.5, lat0_range - 30)

pressure_level = 500
file_path_pattern = '/data/deluge/reanalysis/REANALYSIS/ERA5/3D/4xdaily/hgt/hgt.*.nc'
all_files = glob.glob(file_path_pattern)
filtered_files = [f for f in all_files if 'hgt.195001' <= f.split('/')[-1] <= 'hgt.202212.nc']
filtered_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[1]))

output_temp_dir = "/share/data1/Students/jfields/BlockingDataset/temp_1d_blocks"
os.makedirs(output_temp_dir, exist_ok=True)

output_temp_files = []

for file_path in filtered_files:
    ds = xr.open_dataset(file_path, chunks='auto')
    ds_500 = ds.sel(level=pressure_level)

    ds_daily = ds_500.resample(time='1D').mean()
    ds_daily = ds_daily.drop_vars('level')

    Z500_lat0 = ds_daily['hgt'].sel(latitude=lat0_range, method='nearest')
    Z500_latS = ds_daily['hgt'].sel(latitude=latS_range, method='nearest')
    Z500_latN = ds_daily['hgt'].sel(latitude=latN_range, method='nearest')
    Z500_lat15S = ds_daily['hgt'].sel(latitude=lat15S_range, method='nearest')
    Z500_lat30S = ds_daily['hgt'].sel(latitude=lat30S_range, method='nearest')

    # === Critical calculations (keep .values) ===
    GHGS = (Z500_lat0.values - Z500_latS.values) / 20
    GHGN = (Z500_latN.values - Z500_lat0.values) / 20
    gradient_S = (Z500_lat15S.values - Z500_lat30S.values)

    valid_mask = ((GHGS > 0) & (GHGN < -10) & (gradient_S < 0)).astype(int)
    valid_mask = xr.DataArray(valid_mask, coords=Z500_lat0.coords, dims=Z500_lat0.dims)

    GHGS_da = xr.DataArray(GHGS, coords=Z500_lat0.coords, dims=Z500_lat0.dims)
    GHGN_da = xr.DataArray(GHGN, coords=Z500_lat0.coords, dims=Z500_lat0.dims)

    ds_out = xr.Dataset({
        'block_tag': valid_mask,
        'GHGS': GHGS_da,
        'GHGN': GHGN_da
    })

    # Write out a temporary NetCDF for each file
    year = file_path.split('.')[-2]
    temp_path = os.path.join(output_temp_dir, f'block_tag_1d.{year}.nc')
    ds_out.to_netcdf(temp_path, format='NETCDF4_CLASSIC')
    output_temp_files.append(temp_path)

    print(f'{file_path}')
    ds.close()
    del ds, ds_500, ds_daily, Z500_lat0, Z500_latS, Z500_latN, Z500_lat15S, Z500_lat30S
    del GHGS, GHGN, gradient_S, valid_mask, GHGS_da, GHGN_da, ds_out

# === Final merge of all output files (optional, or do this later) ===
combined = xr.open_mfdataset(output_temp_files, combine='by_coords')

final_dataset = xr.Dataset({
    'block_tag': combined['block_tag'],
    'GHGS': combined['GHGS'],
    'GHGN': combined['GHGN']
})

encoding = {
    'block_tag': {"dtype": "float64", "zlib": True, "complevel": 1},
    'GHGS': {"dtype": "float64", "zlib": True, "complevel": 1},
    'GHGN': {"dtype": "float64", "zlib": True, "complevel": 1},
    'time': {"dtype": "int32", "zlib": True, "complevel": 1, "calendar": "gregorian", "units": "days since 1900-01-01"},
    'latitude': {"dtype": "float64", "zlib": True, "complevel": 1},
    'longitude': {"dtype": "float64", "zlib": True, "complevel": 1},
}

final_dataset = final_dataset.chunk({'time': 365})  # or adjust as needed

final_dataset.to_netcdf(
    '/share/data1/Students/jfields/BlockingDataset/block_tag_1d.19502022.nc',
    format='NETCDF4_CLASSIC',
    engine='netcdf4',
    encoding=encoding,
    compute=True
)
