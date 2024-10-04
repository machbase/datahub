# Required library setup 
import pandas as pd 

# Load data
df = pd.read_csv('./dataset.csv', index_col=0, header=[0, 1]).sort_index(axis=1)

# Modify multi-level column names to single column names
df.columns = [f"{ticker}_{indicator}" for ticker, indicator in df.columns]

# Linear interpolation
df = df.interpolate(method='linear')
# Fill NaN values in the first row with the next value
df = df.fillna(method='bfill')

# Reset the index and convert to time column
df.reset_index(inplace=True)
df.rename(columns={'timestamp': 'time'}, inplace=True)

# Transform the data
df = df.melt(id_vars=['time'], var_name='name', value_name='value')

# Convert the time column to datetime format
df['time'] = pd.to_datetime(df['time'], format='mixed')

# Change the order of the data columns
df = df[['name','time','value']]
df.columns = ['NAME', 'TIME', 'VALUE']

# Convert epoch time to UTC time
epoch = pd.Timestamp('1970-01-01', tz='UTC')

# Change time format
df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S')
df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

# Save the integrated DataFrame
df.to_csv('./datahub-2024-07-SP500.csv', index=False)
