import pandas as pd

# List of files
data = ['./hk1.csv','./hk2.csv','./ss.csv']

# Load data 
df = pd.read_csv(data[0])
df1 = pd.read_csv(data[1])
df2 = pd.read_csv(data[2])

# Modify data columns 
df2.columns = ['time', 'ss_1', 'ss_2', 'ss_3', 'ss_4', 'ss_5', 'ss_6', 'ss_7', 'ss_8', 'ss_9', 'ss_10']

# Transform data
df = df.melt(id_vars=['time'], var_name='name', value_name='value')
df1 = df1.melt(id_vars=['time'], var_name='name', value_name='value')
df2 = df2.melt(id_vars=['time'], var_name='name', value_name='value')

# Merge data
df_ = pd.concat([df, df1, df2], axis=0).reset_index(drop=True)

# Convert 'time' column to datetime format
df_['time'] = pd.to_datetime(df_['time'], format='mixed')

# Change order of data columns
df_ = df_[['name', 'time', 'value']]
df_.columns = ['NAME', 'TIME', 'VALUE']

# Remove missing values
df_ = df_.dropna().reset_index(drop=True)

# Convert epoch time to UTC time 
epoch = pd.Timestamp('1970-01-01', tz='UTC')

# Change time format
df_['TIME'] = pd.to_datetime(df_['TIME'], format='%Y-%m-%d %H:%M:%S')
df_['TIME'] = df_['TIME'].dt.tz_localize('Asia/Seoul')
df_['TIME'] = df_['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

# Save the consolidated DataFrame
df_.to_csv('./datahub-2024-04-wind_elec_gen1.csv', index=False)