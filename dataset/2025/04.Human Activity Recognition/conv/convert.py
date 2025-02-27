# Import necessary libraries
import pandas as pd

# data load
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

# concat train & test dataset
df_total = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# Data shuffle
df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)   

# Set arbitrary start time
start_time = pd.Timestamp("2025-02-24 00:00:00")

# Generate timestamps with a 1-minute interval
timestamps = pd.date_range(start=start_time, periods=6461328, freq="1T")

# Set TIME column
df_total['TIME'] = timestamps

# Move TIME column to the front
df_total = df_total[['TIME'] + [col for col in df_total.columns if col != 'TIME']]

# Convert label values

# Define mapping dictionary
mapping = {v: i for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 130.0, 140.0])}

# Apply mapping
df_total['Activity'] = df_total['Activity'].map(mapping)

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
df = data_change_db(df_total)

# Save the DataFrame to a CSV file
df.to_csv(f'./datahub-2025-4-human-activity.csv', index=False)