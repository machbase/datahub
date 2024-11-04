# Import necessary libraries
import pandas as pd

# Load data
df = pd.read_csv('./San_Diego_Daily_Weather_Data_2014.csv')

# Rename timestamp column to 'time'
df = df.rename(columns={'hpwren_timestamp': 'time'})
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Remove unnecessary columns
df = df.drop(['rowID'], axis=1)

# Resample data at 1-minute intervals
df = df.resample('1T').mean()

# Fill NaN values using linear interpolation
df = df.interpolate(method='linear', limit_direction='both')
df = df.reset_index()

# Data transformation function
def data_change_db(df):
    
    # Transform data
    df = df.melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Reorder data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert reference time (epoch time) to UTC time
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Apply data transformation function (convert name, time, value format & epoch time)
df = data_change_db(df)

# Save the DataFrame to a CSV file
df.to_csv('./datahub-2024-14-San-Diego-Daily-Weather.csv', index=False)
 