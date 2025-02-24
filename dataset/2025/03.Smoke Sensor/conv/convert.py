# Import necessary libraries
import pandas as pd

# data load
df = pd.read_csv('./smoke_detection_iot.csv', index_col=0)

# Data transformation function
def data_change_db(df):
    
    # Set start time
    start_time = pd.Timestamp("2025-01-16 00:00:00")

    # Generate 62,630 timestamps with a 1-second interval
    timestamps = pd.date_range(start=start_time, periods=62630, freq="1S")

    # Convert to DataFrame
    df['UTC'] = timestamps

    # Rename column
    df.rename(columns={'UTC': 'TIME'}, inplace=True)

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
df.to_csv(f'./datahub-2025-3-Smoke.csv', index=False)