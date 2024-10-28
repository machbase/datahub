# Import necessary libraries
import pandas as pd
import datetime
import os

# Set directory
directory_path = './'

# Create an empty list to store DataFrames
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    
    if filename.endswith('.csv'):

        # set base name 
        base_name = filename[:-4]
        # Load the CSV file
        df = pd.read_csv(os.path.join(directory_path, filename), names = ['time', f'{base_name}_normal', f'{base_name}_type1', f'{base_name}_type2', f'{base_name}_type3'])
        # Transform data
        df = df.melt(id_vars=['time'], var_name='name', value_name='value')
        # Append the DataFrame to the list
        dataframes.append(df)  
        
# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Process files and get results
epoch_2024 = int(datetime.datetime(2024, 1, 1).timestamp())
base_time = epoch_2024 * 1000000000  # Multiply by 1 billion (convert to nano)

# Apply the transformation to the 'time' column
combined_df['time'] = combined_df['time'].apply(lambda x: int(base_time + (float(x) * 1000000000)))

# Save DataFrame
combined_df.to_csv('./datahub-2024-2-rotor.csv', index=False)

