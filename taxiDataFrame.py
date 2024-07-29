import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
    try:
        df = pd.read_csv(file_path, header=None, parse_dates=[1], names=[
                         'taxi_id', 'date_time', 'longitude', 'latitude'])
        D.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Check if D is not empty before concatenation
if D:
    # Concatenate all DataFrames in the list into a single DataFrame
    Data = pd.concat(D, ignore_index=True)
    print(f"Concatenated DataFrame shape: {Data.shape}")
else:
    print("No valid data files found.")
    exit()

# Remove duplicates and handle missing values
Data.drop_duplicates(inplace=True)
Data.dropna(inplace=True)
print(f"Data shape after dropping duplicates and NA: {Data.shape}")

# Use a smaller subset of data
subset_size = 50000  # Adjust this number as needed
Data_subset = Data.sample(n=subset_size, random_state=42)
print(f"Subset Data shape: {Data_subset.shape}")

# Compute time differences between consecutive records for each taxi
Data_subset = Data_subset.sort_values(by='date_time', ascending=True).groupby(
    'taxi_id').diff().dropna()
T = Data_subset['date_time'].dt.total_seconds().div(
    60)  # Convert time delta to minutes

# Prepare longitude and latitude data
lon = Data_subset['longitude']
lat = Data_subset['latitude']

# Combine features into a single DataFrame for clustering
features = pd.DataFrame(
    {'longitude': lon, 'latitude': lat, 'time_interval': T})

# Check for NaN or infinite values before scaling
print(f"Features summary before scaling:\n{features.describe()}")

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Check for NaN or infinite values after scaling
if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
    print("NaN or infinite values found after scaling.")
    exit()

# Apply DBSCAN clustering
eps = 0.5  # You can adjust the epsilon parameter
min_samples = 5  # You can adjust the minimum number of samples per cluster
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
features['cluster'] = dbscan.fit_predict(features_scaled)

# Print the cluster assignments
print(features.head())

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(features['longitude'], features['latitude'],
            c=features['cluster'], cmap='viridis', marker='.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering of Taxi Pickups')
plt.colorbar(label='Cluster')
plt.show()

# Optional: Save the clustered data to a CSV file
features.to_csv('clustered_taxi_pickups.csv', index=False)
