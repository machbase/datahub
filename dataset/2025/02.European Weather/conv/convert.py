# Import necessary libraries
import pandas as pd 

# data load
df = pd.read_csv('./weather_data.csv')

# Convert the format of the 'timestamp' column
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

# Rename column
df.rename(columns={'utc_timestamp': 'TIME'}, inplace=True)

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
    epoch = pd.Timestamp('1970-01-01')

    # Change TIME format 
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Apply data transformation function (convert name, time, value format & epoch time)
df = data_change_db(df)

# Save the DataFrame to a CSV file
df.to_csv(f'./datahub-2025-2-EU-weather.csv', index=False)