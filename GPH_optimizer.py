import glob
import os
import xarray as xr
import logging

#########################################################
# REQUIRED USER INPUTS:

# Line 16: Define folder path for input files
# Line 17: Define output file name
# Line 27: Select time range
#########################################################

####### Load input files #######
pressure_level = 500
file_path_pattern = '/folder_path/hgt.*.nc'
final_output_filename = 'ERA5_hgt_500mb.19502024_optimized.nc'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

####### Find and sort input files ######
all_files = glob.glob(file_path_pattern)
if not all_files:
    raise FileNotFoundError(f"No files found matching pattern: {file_path_pattern}")

filtered_files = [f for f in all_files if 'hgt.195001' <= os.path.basename(f) <= 'hgt.202412.nc']
filtered_files.sort(key=lambda f: int(os.path.basename(f).split('.')[1]))

if not filtered_files:
    raise ValueError("No files found in the specified date range.")

logging.info(f"Found {len(filtered_files)} files to process.")

####### Process each file: select 500 hPa and Northern Hemisphere #######
ds_list = []

for file in filtered_files:
    logging.info(f"Processing file: {file}")
    try:
        ds = xr.open_dataset(file)
        ds_500 = ds.sel(level=pressure_level, method='nearest')
        ds_500_nh = ds_500.sel(latitude=slice(90, 0))
        ds_list.append(ds_500_nh)
    except Exception as e:
        logging.warning(f"Error processing file {file}: {e}")

####### Concatenate datasets along time dimension #######
logging.info("Concatenating datasets...")
try:
    ds_combined = xr.concat(ds_list, dim='time')
    logging.info("Concatenation complete.")
except Exception as e:
    raise RuntimeError(f"Error concatenating datasets: {e}")

####### Resample to daily mean using Dask for efficiency #######
logging.info("Opening dataset with Dask for resampling...")
ds_combined = ds_combined.chunk({'time': 100})
logging.info("Resampling to daily means...")
ds_daily = ds_combined.resample(time='1D').mean()

####### Save final daily-mean dataset #######
logging.info(f"Saving final dataset to {final_output_filename}...")
try:
    ds_daily.to_netcdf(final_output_filename)
    logging.info("Final file saved successfully.")
except Exception as e:
    raise RuntimeError(f"Error saving data to {final_output_filename}: {e}")
