# Required libraries
import pandas as pd 

# Load data
df = pd.read_csv('./energydata_complete.csv')

# Convert the 'date' column to 'time'
df.rename(columns={'date': 'time'}, inplace=True)

# Reshape the data
df = df.melt(id_vars=['time'], var_name='name', value_name='value')

# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='mixed')

# Change the column order
df = df[['name','time','value']]
df.columns = ['NAME', 'TIME', 'VALUE']

# Convert reference time (epoch time) to UTC time
epoch = pd.Timestamp('1970-01-01', tz='UTC')

# Change time format
df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S')
df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

# Save dataframe
df.to_csv('./datahub-2024-09-Appliances-Energy.csv', index=False)