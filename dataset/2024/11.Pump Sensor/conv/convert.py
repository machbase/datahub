# Import necessary libraries
import pandas as pd

# Load the data
df = pd.read_csv('./sensor.csv', index_col=0)

# Change 'Normal' to 0, and others to 1
# The reason for changing others to 1 is that the number of 'Broken' cases is small and is included in the 'Recovering' category
df['machine_status'] = df['machine_status'].replace({'NORMAL': 0, 'RECOVERING': 1, 'BROKEN': 1})

# Calculate the number of null values in each column
null_counts = df.isnull().sum()

# Select columns that have 1000 or more null values
df_null = df.loc[:, null_counts >= 1000]

# Remove columns that have 1000 or more null values
df = df.drop(df_null, axis=1)

# Fill missing values (fewer than 1000) with the previous value
df = df.fillna(method='ffill')

# Rename the 'timestamp' column to 'time'
df = df.rename(columns={'timestamp': 'time'})

# Data transformation function
def data_change_db(df):
    
    # Data transformation
    df = df.melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert the 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the column order
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert reference time (epoch time) -> UTC time
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Data transformation
df = data_change_db(df)

# Save the DataFrame
df.to_csv('./datahub-2024-11-Pump-Sensor.csv', index=False)


