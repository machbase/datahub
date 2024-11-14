# Import necessary libraries
import pandas as pd
import os
from datetime import datetime, timedelta

# Load metadata
df = pd.read_csv('./cleaned_dataset/metadata.csv')

# Use only discharge data and select the battery filename and capacity columns
df = df[df['type'] == 'discharge'][['filename', 'Capacity']]

# Data transformation function
def data_change_db(df):
    
    # Set the base time (2010-07-21 15:00:00.000)
    base_time = datetime(2010, 7, 21, 15, 0, 0)

    # Convert the 'Time' column to relative time based on the base time
    df['Time'] = df['Time'].apply(lambda x: base_time + timedelta(seconds=x))

    # Reshape the data
    df = df.melt(id_vars=['Time'], var_name='name', value_name='value')

    # Change the order of the columns
    df = df[['name', 'Time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']
    
    # Convert the base time (epoch time) to UTC time
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change the time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='mixed')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Create a list of dataframes with capacity added to each file
merged_df = []

for index, row in df.iterrows():
    filename = row['filename']
    capacity = row['Capacity']
    
    # Extract the prefix of the filename (before the .csv)
    file_prefix = os.path.splitext(filename)[0]
    
    # Read the CSV file
    try:
        df = pd.read_csv('./cleaned_dataset/data/' + filename)
        # Add the capacity column
        df['Capacity'] = capacity
        # Add the filename prefix to the column names
        df.columns = [f"{'B'+ file_prefix[1:]}_{col}" if col != "Time" else col for col in df.columns]
        # Convert the data format
        df = data_change_db(df)
        merged_df.append(df)
    except FileNotFoundError:
        print(f"File {filename} not found.")

# Create a merged dataframe with all the data
merged_df = pd.concat(merged_df, ignore_index=True)

# Remove rows where the 'VALUE' column is equal to '[]'
merged_df = merged_df[merged_df['VALUE'] != '[]']

# Save the dataframe
merged_df.to_csv('./datahub-2024-16-NASA-Lithium_battery.csv', index=False) 