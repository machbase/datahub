# Import necessary libraries
import pandas as pd
import glob
import os

# Data transformation function
def data_change_db(df):
    
    # Data transformation
    df = df.melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the order of the data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert epoch time to UTC time 
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Set the folder path where the files are located
folder_path = './data/'

# Find all CSV files and list them
file_list = glob.glob(os.path.join(folder_path, '*.csv'))

# Create a list to hold dataframes
data_frames = []

for file_path in file_list:
    # Extract the file name part (excluding the extension) from the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Read the file
    df = pd.read_csv(file_path)
    
    # Fill NaN values using linear interpolation
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Remove leading and trailing missing values
    df = df.dropna()
    
    # Prefix each column name with the file name
    df = df.add_prefix(f"{file_name}_")
    
    # Remove the end time column
    df = df.drop([f"{file_name}_To Date"], axis=1)
    
    # Rename the start time column to 'time'
    df = df.rename(columns={f"{file_name}_From Date": 'time'}) 
    
    # Apply data transformation function (convert name, time, value format & epoch time)
    df = data_change_db(df)
    
    # Add the dataframe to the list
    data_frames.append(df)
    
# Merge all dataframes into one dataframe along the row axis
merged_df = pd.concat(data_frames, axis=0).reset_index(drop=True)

# Save the DataFrame 
merged_df.to_csv('./datahub-2024-13-India-Air-Quality.csv', index=False)   
