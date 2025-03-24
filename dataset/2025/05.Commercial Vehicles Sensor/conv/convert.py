# Import necessary libraries
import pandas as pd

# data load
df = pd.read_csv('./Terra-D2-multi-labeled-interpolated.csv')

# Set base time (midnight on January 1st, 2025)
base_time = pd.to_datetime("2025-01-01 00:00:00")

# Convert seconds to datetime
converted_times = df['time'].apply(lambda x: base_time + pd.to_timedelta(x, unit='s'))

# Update the 'time' column with the converted values
df['time'] = converted_times

# Rename the 'time' column to 'TIME'
df.rename(columns={'time': 'TIME'}, inplace=True)

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

# Apply data transformation function
df = data_change_db(df)

# Save the DataFrame to a CSV file
df.to_csv(f'./datahub-2025-5-commercial-vehicles.csv', index=False)