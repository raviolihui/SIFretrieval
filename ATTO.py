import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import numpy as np

# Path to the folder containing the NetCDF files
data_path = "/home/carmen/Documents/masterthesis/SIF_TEST_DATA/2023-m/"

# ATTO coordinates
atto_lat = -2.1433
atto_lon = -59.0004

# Create an empty list to store monthly SIF values
monthly_sif = []

# List all files in the directory
files = os.listdir(data_path)

# Sort the files to ensure correct order
files.sort()

# Loop through the months (1 to 12 for 2023)
for month in range(1, 13):
    # Define the month format to match in filenames, e.g., "202301" for January
    month_str = f"2023{month:02d}"

    # Find the files corresponding to the current month
    month_files = [file for file in files if month_str in file]

    if month_files:
        # Process the first file for the month (assuming only one file per month)
        file_name = month_files[0]
        file_path = os.path.join(data_path, file_name)

        # Load the NetCDF file
        ds = xr.open_dataset(file_path)

        # Extract SIF for ATTO coordinates
        sif = ds['solar_induced_fluorescence'].sel(
            latitude=atto_lat, longitude=atto_lon, method='nearest'
        ).values

        # Append the SIF value to the list
        monthly_sif.append(sif)


# Define datetime values for the months (Jan to Dec 2023)
months = pd.to_datetime([f'2023-{month:02d}-01' for month in range(1, 13)])

# Create the pandas DataFrame
time_series = pd.DataFrame({
    'Month': months,
    'SIF': monthly_sif
})

# Define datetime values for the months (Jan to Dec 2023)
months = pd.to_datetime([f'2023-{month:02d}-01' for month in range(1, 13)])

# Plot the time series directly
plt.plot(months, monthly_sif, marker='o', linestyle='-', color='b')
plt.title('Monthly SIF for ATTO Tower (2023)')
plt.xlabel('Month')
plt.ylabel('SIF')
plt.xticks(months, [f'{month.month:02d}' for month in months])  # Format x-ticks to show just month numbers
plt.grid(True)
plt.show()

# Save the plot to a file
plt.savefig("/home/carmen/Documents/masterthesis/SIF_TEST_DATA/atto_sif_time_series_2023.png")
plt.close()  # Close the plot to free up memory