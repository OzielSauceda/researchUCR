import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

Data = pd.DataFrame()
F = []

dataset_path = r'C:/Users/bosso/OneDrive/Desktop/REU Research/Taxi Pickups/release/taxi_log_2008_by_id'

for root, dirs, files in os.walk(dataset_path):
    for name in files:
        F.append(os.path.join(root, name))

D = []
for index, file_path in enumerate(F):
    try:
        df = pd.read_csv(file_path, header=None, parse_dates=[1], names=[
                         'taxi_id', 'date_time', 'longitude', 'latitude'])
        D.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if D:
    Data = pd.concat(D, ignore_index=True)
    print(f"Concatenated DataFrame shape: {Data.shape}")
else:
    print("No valid data files found.")
    exit()

Data.drop_duplicates(inplace=True)
Data.dropna(inplace=True)
print(f"Data shape after dropping duplicates and NA: {Data.shape}")

subset_size = 20000
Data_subset = Data.sample(n=subset_size, random_state=42)
print(f"Subset Data shape: {Data_subset.shape}")

Data_subset = Data_subset.sort_values(
    by='date_time', ascending=True).groupby('taxi_id').diff().dropna()
T = Data_subset['date_time'].dt.total_seconds().div(60)

lon = Data_subset['longitude']
lat = Data_subset['latitude']

features = pd.DataFrame(
    {'longitude': lon, 'latitude': lat, 'time_interval': T})
print(f"Features summary before scaling:\n{features.describe()}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
    print("NaN or infinite values found after scaling.")
    exit()

num_clusters = 5
hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
features['cluster'] = hierarchical.fit_predict(features_scaled)

print(features.head())

plt.ion()

plt.figure(figsize=(10, 6))
plt.scatter(features['longitude'], features['latitude'],
            c=features['cluster'], cmap='viridis', marker='.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Hierarchical Clustering of Taxi Pickups')
plt.colorbar(label='Cluster')
plt.show()

features.to_csv('clustered_taxi_pickups.csv', index=False)

plt.ioff()
plt.show()
