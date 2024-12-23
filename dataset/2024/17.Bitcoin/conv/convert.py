# Import necessary libraries
import pandas as pd
import os
from datetime import datetime, timedelta

# Load metadata
df = pd.read_csv('./Bitcoin.csv')

# Data transformation function
def data_change_db(df):
    
    # Data transformation
    df = df.melt(id_vars=['TIME'], var_name='name', value_name='value')

    # Convert 'TIME' column to dateTIME format
    df['TIME'] = pd.to_datetime(df['TIME'], format='mixed')

    # Change the order of the data columns
    df = df[['name', 'TIME', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert epoch TIME to UTC TIME 
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change TIME format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Apply data transformation function (convert name, time, value format & epoch time)
df = data_change_db(df)

# Remove NaN Values
df = df.dropna().reset_index(drop=True)

# Save the DataFrame to a CSV file
df.to_csv('./datahub-2024-17-Bitcoin.csv', index=False)