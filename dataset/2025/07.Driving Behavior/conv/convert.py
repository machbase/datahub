# Import necessary libraries
import pandas as pd

# data load
df_train = pd.read_csv('./train_motion_data.csv')
df_test = pd.read_csv('./test_motion_data.csv')

# Set label to int
label_map = {'NORMAL': 0, 'SLOW': 1, 'AGGRESSIVE': 2}

for df in [df_train, df_test]:
    df['Class'] = df['Class'].map(label_map)

# concat train and test data
df = pd.concat([df_train, df_test], axis=0)

# Rename the 'time' column to 'TIME'
df.rename(columns={'Timestamp': 'TIME'}, inplace=True)

# Move the 'TIME' column to the front
cols = ['TIME'] + [col for col in df.columns if col != 'TIME']
df = df[cols]

# Data transformation function
def data_change_db(df):
    
    # Set start time
    start_time = pd.Timestamp("2025-07-18 00:00:00")

    # Generate 62,630 timestamps with a 1-second interval
    timestamps = pd.date_range(start=start_time, periods=6728, freq="1S")

    # Convert to DataFrame
    df['TIME'] = timestamps

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
df.to_csv(f'./datahub-2025-7-Driving-Behavior.csv', index=False)