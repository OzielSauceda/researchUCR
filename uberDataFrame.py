import pandas as pd
import os
import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# DBSCAN clustering

# Set the folder pathway
path = r'C:/Users/bosso/OneDrive/Desktop/REU Research/Uber Pickups'

# Use glob to get all the csv files in the folder
csv_files = glob.glob(os.path.join(path, "*.csv"))

# Initialize an empty list to store DataFrames
dfs = []

# Loop over the list of csv files
for f in csv_files:
    try:
        # Read the csv file
        df = pd.read_csv(f, encoding='ISO-8859-1')

        # Print the location and filename
        print('Location:', f)
        print('File Name:', os.path.basename(f))

        # Print the columns present in the file
        print('Columns:', df.columns.tolist())

        # Print the content
        print('Content:')
        print(df.head())  # Print only the first few rows for brevity
        print()

        # Print the shape of the dataframe
        print('Shape of DataFrame:', df.shape)
        print()

        # Append the DataFrame to the list
        dfs.append(df)

    except Exception as e:
        print(f"Error reading {f}: {e}")

# Concatenate all DataFrames into a single DataFrame, handling different columns
combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# Print the shape of the combined dataframe
print('Shape of Combined DataFrame:', combined_df.shape)

# Specify the output file path
output_file = r'C:/Users/bosso/OneDrive/Desktop/combined_uber_pickups.csv'

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(output_file, index=False)

# Check if 'Lat' and 'Lon' columns exist and have valid data
if 'Lat' in combined_df.columns and 'Lon' in combined_df.columns:
    combined_df = combined_df.dropna(subset=['Lat', 'Lon'])
    if not combined_df.empty:
        # Use a smaller subset of data
        subset_size = 100000  # Adjust this number as needed
        combined_df_subset = combined_df.sample(n=subset_size, random_state=42)
        print(f"Subset Data shape: {combined_df_subset.shape}")

        # Select features for clustering
        features = ['Lat', 'Lon']
        data = combined_df_subset[features]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Apply DBSCAN clustering
        eps = 0.1  # You can adjust the epsilon parameter
        min_samples = 10  # You can adjust the minimum number of samples per cluster
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        combined_df_subset['Cluster'] = dbscan.fit_predict(scaled_data)

        # Specify the output file path for the clustered DataFrame
        clustered_output_file = r'C:/Users/bosso/OneDrive/Desktop/combined_uber_pickups_with_clusters.csv'

        # Save the DataFrame with cluster labels to a new CSV file
        combined_df_subset.to_csv(clustered_output_file, index=False)

        # Analyze and visualize the clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(combined_df_subset['Lon'], combined_df_subset['Lat'],
                    c=combined_df_subset['Cluster'], cmap='viridis')
        plt.colorbar(label='Cluster')
        plt.title('DBSCAN Clustering of Uber Pickups')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    else:
        print("No valid data in 'Lat' and 'Lon' columns after dropping NaNs.")
else:
    print("'Lat' and/or 'Lon' columns are not present in the combined DataFrame.")
