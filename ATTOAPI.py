import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Base URL for the data
url_base = "https://radiantearth.github.io/stac-browser/#/external/data-portal.s5p-pal.com/api/s5p-l3/sif/day/2023/s5p-l3grd-sif-001-day-"
https://data-portal.s5p-pal.com/download/s5p-l3/5c12b532-7df6-4f95-ab61-3a46d2321d72
https://data-portal.s5p-pal.com/download/s5p-l3/dceb3ea5-0e2b-4b8d-b2fe-a5c437b47d00
# Directory to save data
data_path = "/home/carmen/Documents/masterthesis/SIF_2023/"
os.makedirs(data_path, exist_ok=True)


# Function to download SIF data for each day
def download_data(date):
    date_str = date.strftime('%Y%m%d')  # Format the date as YYYYMMDD
    file_url = f"{url_base}{date_str}-20240325.json"
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Will raise an HTTPError if status code is not 200
        # Save the JSON response to a file
        file_name = f"s5p-l3grd-sif-001-day-{date_str}-20240325.json"
        file_path = os.path.join(data_path, file_name)

        with open(file_path, 'w') as f:
            f.write(response.text)
        print(f"Data for {date_str} downloaded successfully.")
        return file_path
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch data for {date_str}: {e}")
        return None


# Generate a list of all dates in 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
delta = timedelta(days=1)

current_date = start_date
all_files = []

# Loop through each date in 2023 and download data
while current_date <= end_date:
    file_path = download_data(current_date)
    if file_path:
        all_files.append(file_path)
    current_date += delta  # Move to the next day


# Now that data is downloaded, you can process the downloaded JSON files into a time series
def extract_sif_values(files):
    sif_values = []
    dates = []

    for file in files:
        with open(file, 'r') as f:
            data = requests.get(file).json()
            # Adjust this to reflect the actual structure of your JSON data
            sif_value = data.get('value', None)  # Modify based on actual key for SIF
            if sif_value is not None:
                # Extract the date from the filename (YYYYMMDD)
                date_str = os.path.basename(file).split('-')[3]
                dates.append(pd.to_datetime(date_str))
                sif_values.append(sif_value)
            else:
                print(f"No SIF value found for {file}")

    # Create the time series DataFrame
    time_series = pd.DataFrame({
        'Date': dates,
        'SIF': sif_values
    })

    return time_series


# Extract the time series
time_series = extract_sif_values(all_files)

# Now you can plot the time series or save it
time_series.to_csv("/home/carmen/Documents/masterthesis/atto_sif_time_series_2023.csv", index=False)
print("Time series saved.")
