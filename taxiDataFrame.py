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

# Sample a subset of the data to reduce memory usage
sample_fraction = 0.1  # Use 10% of the data, adjust as needed
Data_sampled = Data.sample(frac=sample_fraction, random_state=42)
print(f"Sampled Data shape: {Data_sampled.shape}")

# Compute time differences between consecutive records for each taxi
Data_sampled = Data_sampled.sort_values(by='date_time', ascending=True).groupby(
    'taxi_id').diff().dropna()
T = Data_sampled['date_time'].dt.total_seconds().div(
    60)  # Convert time delta to minutes

# Prepare longitude and latitude data
lon = Data_sampled['longitude']
lat = Data_sampled['latitude']

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
eps_value = 0.5  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
# Number of samples (or total weight) in a neighborhood for a point to be considered as a core point
min_samples_value = 5

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
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
features.to_csv('dbscan_clustered_taxi_pickups_sampled.csv', index=False)
