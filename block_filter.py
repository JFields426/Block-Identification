import numpy as np
import xarray as xr
from netCDF4 import Dataset

#########################################################
# REQUIRED USER INPUTS:

# [OPTIONAL] Line 14: Change input file (if needed)
# [OPTIONAL] Line 15: Change output file
# [OPTIONAL] Line 52: Change block area thresholds
#########################################################

####### Define input and output file names #######
input_file = "block_id.19502024.nc"
output_file = "filtered_block_id.19502024.nc"

####### Load and filter by block area #######
print("Loading data and filtering by block area...")

# Load dataset
ds = xr.open_dataset(input_file)
obj_id = ds["object_id"]
lat = ds["latitude"].values
lon = ds["longitude"].values

# Earth's radius in km
R = 6371.0

# Convert lat/lon to radians
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Compute grid spacing
dlat = np.abs(np.gradient(lat_rad))
dlon = np.abs(np.gradient(lon_rad))

# Compute grid cell area in km²
cos_lat = np.cos(lat_rad[:, np.newaxis])
area_km2 = dlat[:, np.newaxis] * dlon[np.newaxis, :] * cos_lat * R**2

# Track block areas
block_areas = {}
for t in range(obj_id.sizes['time']):
    ids = obj_id[t].values
    unique_ids = np.unique(ids[ids != 0])
    for bid in unique_ids:
        area = np.sum(area_km2[ids == bid])
        if bid not in block_areas or area > block_areas[bid]:
            block_areas[bid] = area

# Keep blocks with area between 2.0e6 and 7.5e6 km²
valid_ids = {bid for bid, area in block_areas.items() if 2.0e6 <= area <= 7.5e6}
print(f"Number of valid block IDs: {len(valid_ids)}")

# Build sequential ID mapping
valid_ids_sorted = sorted(valid_ids)
id_mapping = {old: new for new, old in enumerate(valid_ids_sorted, start=1)}
print("Mapping of original to sequential IDs (excluding 0):")
for old, new in id_mapping.items():
    print(f"{old} -> {new}")

####### Create output NetCDF with remapped IDs #######
print("Writing filtered and remapped data to output file...")

# Open new NetCDF file
dst_nc = Dataset(output_file, 'w', format='NETCDF4')

# Copy dimensions
for dim_name, dim in ds.dims.items():
    dst_nc.createDimension(dim_name, len(ds[dim]))

# Copy coordinate variables
for var_name in ["latitude", "longitude", "time"]:
    varin = ds[var_name]
    var_out = dst_nc.createVariable(var_name, varin.dtype, varin.dims)
    var_out.setncatts({k: varin.attrs[k] for k in varin.attrs})
    var_out[:] = varin[:]

# Create the object_id variable
fill_value = obj_id.attrs.get("_FillValue", 0)
obj_out = dst_nc.createVariable("object_id", obj_id.dtype, obj_id.dims, fill_value=fill_value)
obj_out.setncatts({k: v for k, v in obj_id.attrs.items() if k != "_FillValue"})

# Apply filtering and remapping
remap_func = np.vectorize(lambda x: id_mapping.get(x, 0))  # Map invalid IDs to 0

for t in range(obj_id.sizes['time']):
    block_slice = obj_id[t].values
    block_slice[~np.isin(block_slice, list(valid_ids))] = 0
    remapped = remap_func(block_slice)
    obj_out[t, :, :] = remapped
    if t % 100 == 0:
        print(f"Processed time step {t + 1} / {obj_id.sizes['time']}")

# Close output file
dst_nc.close()
print(f"Final file with filtered sequential block IDs saved as '{output_file}'")
