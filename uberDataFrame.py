# import necessary libraries
import pandas as pd
import os
import glob

# set the folder pathway
path = r'C:/Users/bosso/OneDrive/Desktop/REU Research/Uber Pickups'

# use glob to get all the csv files in the folder
csv_files = glob.glob(os.path.join(path, "*.csv"))

# Initialize an empty list to store DataFrames
dfs = []

# loop over the list of csv files
for f in csv_files:
    try:
        # read the csv file
        df = pd.read_csv(f, encoding='ISO-8859-1')

        # print the location and filename
        print('Location:', f)
        print('File Name:', os.path.basename(f))

        # print the columns present in the file
        print('Columns:', df.columns.tolist())

        # print the content
        print('Content:')
        print(df.head())  # Print only the first few rows for brevity
        print()

        # print the shape of the dataframe
        print('Shape of DataFrame:', df.shape)
        print()

        # Append the DataFrame to the list
        dfs.append(df)

    except Exception as e:
        print(f"Error reading {f}: {e}")

# Concatenate all DataFrames into a single DataFrame, handling different columns
combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# print the shape of the combined dataframe
print('Shape of Combined DataFrame:', combined_df.shape)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_uber_pickups.csv', index=False)
