import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Initialize an empty DataFrame
Data = pd.DataFrame()
F = []

# Define the dataset path
dataset_path = r'C:/Users/bosso/OneDrive/Desktop/REU Research/Taxi Pickups/release/taxi_log_2008_by_id'

# Walk through the dataset directory and collect all file paths
for root, dirs, files in os.walk(dataset_path):
    for name in files:
        F.append(os.path.join(root, name))

# Read each CSV file into a DataFrame and store them in a list
D = []
for index, file_path in enumerate(F):
    D.append(pd.read_csv(file_path, header=None, parse_dates=[1],
                         names=['taxi_id', 'date_time', 'longitude', 'latitude']))

# Concatenate all DataFrames in the list into a single DataFrame
Data = pd.concat(D, ignore_index=True)

# Remove duplicates and handle missing values
Data.drop_duplicates(inplace=True)
Data.dropna(inplace=True)

# Display the shape of the DataFrame
print(Data.shape)

# Compute time differences between consecutive records for each taxi
D_diff = Data.sort_values(
    by='date_time', ascending=True).groupby('taxi_id').diff()
D_diff.dropna(inplace=True)
T = D_diff.iloc[:, 0]
T /= np.timedelta64(1, 's')
print('Average time interval: ', T[T < 1e3].mean(), 'Sec')
T /= 60

# Prepare longitude and latitude data for distance calculation
lon = Data['longitude'].to_numpy() * np.pi / 180
lat = Data['latitude'].to_numpy() * np.pi / 180
lon1 = lon[:-1]
lon2 = lon[1:]
lat1 = lat[:-1]
lat2 = lat[1:]
Delta_lat = lat2 - lat1
Delta_lon = lon2 - lon1
a = (np.sin(Delta_lat / 2)) ** 2 + np.cos(lat1) * \
    np.cos(lat2) * (np.sin(Delta_lon / 2)) ** 2
Distance = 6371 * 1000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
print('Average distance interval: ', Distance[Distance < 1e5].mean(), 'meters')

# Plot histograms
T_Interval = T
fig = plt.figure(figsize=(10, 4), dpi=100)  # Adjusted figsize and dpi
ax1 = plt.subplot2grid((1, 2), (0, 0))
plt.hist(T_Interval[(T_Interval > 0.2) & (T_Interval < 12)], bins=24, rwidth=0.8, color='r',
         weights=np.ones_like(T_Interval[(T_Interval > 0.2) & (T_Interval < 12)]) / T_Interval[(T_Interval > 0.2) & (T_Interval < 12)].size)
plt.ylabel('proportion', fontsize=10)
plt.text(5.2, -0.06, 'minutes', fontsize=8)
plt.text(3.8, -0.1, '(a) Time Intervals', fontsize=10)
ax2 = plt.subplot2grid((1, 2), (0, 1))
plt.hist(Distance[(Distance < 8000) & (Distance > 250)], bins=16, rwidth=0.8, color='r',
         weights=np.ones_like(Distance[(Distance < 8000) & (Distance > 250)]) / Distance[(Distance < 8000) & (Distance > 250)].size)
plt.text(3500, -0.025, 'meters', fontsize=8)
plt.text(2400, -0.038, '(b) Distance Intervals', fontsize=10)
plt.ylabel('proportion', fontsize=10)
plt.show()

# Filter data for Beijing
Beijing = Data[(116.05 < Data.longitude) & (Data.longitude < 116.8)
               & (39.5 < Data.latitude) & (Data.latitude < 40.25)]

# Plot hexbin maps for Beijing data
plt.figure(figsize=(10, 4), dpi=100)  # Adjusted figsize and dpi
ax1 = plt.subplot2grid((1, 2), (0, 0))
plt.hexbin(Beijing.longitude, Beijing.latitude,
           bins='log', gridsize=600, cmap=plt.cm.hot)
plt.axis([116.05, 116.8, 39.5, 40.25])
plt.title("(a) Data overview in Beijing", fontsize=10)
cb = plt.colorbar()
cb.set_label('log10(N)', fontsize=8)

ax2 = plt.subplot2grid((1, 2), (0, 1))
Ring_Road = Data[(116.17 < Data.longitude) & (Data.longitude < 116.57) & (
    39.76 < Data.latitude) & (Data.latitude < 40.09)]
plt.hexbin(Ring_Road.longitude, Ring_Road.latitude,
           bins='log', gridsize=600, cmap=plt.cm.hot)
plt.axis([116.17, 116.57, 39.76, 40.09])
plt.title("(b) Within the 5th Ring Road of Beijing", fontsize=10)
cb = plt.colorbar()
cb.set_label('log10(N)', fontsize=8)

plt.show()
