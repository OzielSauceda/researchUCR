import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

path = r'C:/Users/bosso/OneDrive/Desktop/REU Research/Uber Pickups'
csv_files = glob.glob(os.path.join(path, "*.csv"))
dfs = []

for f in csv_files:
    try:
        df = pd.read_csv(f, encoding='ISO-8859-1')
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

combined_df = pd.concat(dfs, ignore_index=True, sort=False)
print('Shape of Combined DataFrame:', combined_df.shape)

output_file = r'C:/Users/bosso/OneDrive/Desktop/combined_uber_pickups.csv'
combined_df.to_csv(output_file, index=False)

if 'Lat' in combined_df.columns and 'Lon' in combined_df.columns:
    combined_df = combined_df.dropna(subset=['Lat', 'Lon'])
    if not combined_df.empty:
        subset_size = 10000
        combined_df_subset = combined_df.sample(n=subset_size, random_state=42)
        print(f"Subset Data shape: {combined_df_subset.shape}")

        features = ['Lat', 'Lon']
        data = combined_df_subset[features]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        num_clusters = 5
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
        combined_df_subset['Cluster'] = hierarchical.fit_predict(scaled_data)

        clustered_output_file = r'C:/Users/bosso/OneDrive/Desktop/combined_uber_pickups_with_clusters.csv'
        combined_df_subset.to_csv(clustered_output_file, index=False)

        plt.figure(figsize=(10, 6))
        plt.scatter(combined_df_subset['Lon'], combined_df_subset['Lat'],
                    c=combined_df_subset['Cluster'], cmap='viridis')
        plt.colorbar(label='Cluster')
        plt.title('Hierarchical Clustering of Uber Pickups')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    else:
        print("No valid data in 'Lat' and 'Lon' columns after dropping NaNs.")
else:
    print("'Lat' and/or 'Lon' columns are not present in the combined DataFrame.")
