import xarray as xr
import numpy as np
import os
import re
import numpy as np
import numpy as np





lat_min, lat_max = -30, 15
lon_min, lon_max = -75, -50

sif_amazon = []
time_list = []

data_dir = data_dir = "/Users/carmenoliver/Desktop/SIFRETRIEVAL/DATA/SIF_DATA_TROPOMI/"

files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
print("Files found:", files)
for file in files:
    match = re.search(r"month-(\d{6})", file)
    if match:
        date_str = match.group(1)  
        year = int(date_str[:4])  
        month = int(date_str[4:6])  

        file_path = os.path.join(data_dir, file)
        
        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            continue
        ds_amazon = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        
        sif_amazon.append(ds_amazon["solar_induced_fluorescence"].values)  
        time_list.append(f"{year}-{month:02d}")

if not sif_amazon or not time_list:
    raise ValueError("No valid data found. Check the input files and filtering logic.")
sif_amazon = np.stack(sif_amazon, axis=0).squeeze()  # Shape: [N_months, lat, lon]

years = sorted(set(int(t.split("-")[0]) for t in time_list))
num_years = len(years)

sif_monthly = np.full((num_years, 12, *sif_amazon.shape[1:]), np.nan)
for i, month_data in enumerate(sif_amazon):
    year_idx = i // 12  
    month_idx = i % 12  
    sif_monthly[year_idx, month_idx] = month_data 

print("Fixed sif_monthly shape:", sif_monthly.shape)




year_idx_2024 = years.index(2024)
month_idx_jan = 0  # January

lat_grid = np.linspace(lat_min, lat_max, sif_monthly.shape[2])
lon_grid = np.linspace(lon_min, lon_max, sif_monthly.shape[3])

lat_mask = (lat_grid >= -25) & (lat_grid <= 5)
lon_mask = (lon_grid >= -65) & (lon_grid <= -60)

sif_jan_2024_region = sif_monthly[year_idx_2024, month_idx_jan][np.ix_(lat_mask, lon_mask)]
mean_sif_jan_2024_region = np.nanmean(sif_jan_2024_region)
median_sif_jan_2024_region = np.nanmedian(sif_jan_2024_region)
print("Mean SIF in January 2024 over lat -25 to 5, lon -65 to -60:", mean_sif_jan_2024_region)
print("median:", median_sif_jan_2024_region)