import pandas as pd

# Load data
df1 = pd.read_csv('weather_forecast_hankyeong_4hours.csv')
df2 = pd.read_csv('weather_forecast_seongsan_4hours.csv')

# Set data columns
df1 = df1[['tgt_beg', 'temperature','humidity', 'windspeed', 'winddirection', '6hrain', '6hsnow', 'rainprobability', 'raintype', 'seawave', 'skystatus']]
df1.columns = ['time', 'h_temperature','h_humidity', 'h_windspeed', 'h_winddirection', 'h_6hrain', 'h_6hsnow', 'h_rainprobability', 'h_raintype', 'h_seawave', 'h_skystatus']

df2 = df2[['tgt_beg', 'temperature','humidity', 'windspeed', 'winddirection', '6hrain', '6hsnow', 'rainprobability', 'raintype', 'seawave', 'skystatus']]
df2.columns = ['time', 's_temperature','s_humidity', 's_windspeed', 's_winddirection', 's_6hrain', 's_6hsnow', 's_rainprobability', 's_raintype', 's_seawave', 's_skystatus']

# Transform data
df1 = df1.melt(id_vars=['time'], var_name='name', value_name='value')
df2 = df2.melt(id_vars=['time'], var_name='name', value_name='value')

# Merge data
df_ = pd.concat([df1, df2], axis=0).reset_index(drop=True)

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
df_.to_csv('./datahub-2024-04-wind_elec_gen2.csv', index=False)