# Required library setup 
import pandas as pd 

# Load training data
df_1 = pd.read_csv('./data/0D.csv')
df_2 = pd.read_csv('./data/1D.csv')
df_3 = pd.read_csv('./data/2D.csv')
df_4 = pd.read_csv('./data/3D.csv')
df_5 = pd.read_csv('./data/4D.csv')

# Load test data
df_6 = pd.read_csv('./data/0E.csv')
df_7 = pd.read_csv('./data/1E.csv')
df_8 = pd.read_csv('./data/2E.csv')
df_9 = pd.read_csv('./data/3E.csv')
df_10= pd.read_csv('./data/4E.csv')

# Remove data with mismatched sample rates
df_1 = df_1[df_1['V_in'] != 2.0]
df_2 = df_2[df_2['V_in'] != 2.0]
df_3 = df_3[df_3['V_in'] != 2.0]
df_4 = df_4[df_4['V_in'] != 10.0]
df_5 = df_5[df_5['V_in'] != 10.0]

df_6 = df_6[df_6['V_in'] != 8.1]
df_7 = df_7[df_7['V_in'] != 4.0]
df_8 = df_8[df_8['V_in'] != 8.1]
df_9 = df_9[df_9['V_in'] != 4.0]
df_10 = df_10[df_10['V_in'] != 4.0]

# Create lists of DataFrames
df_list_train = [df_1, df_2, df_3, df_4, df_5]
df_list_test = [df_6, df_7, df_8, df_9, df_10]

# Check if the number of samples divided by the sample rate is 0
for i in range(5):
    print(df_list_train[i].shape[0] // 4096)
    print(df_list_test[i].shape[0] // 4096)
    
# Add an unbalance factor column
df_1['unbalance_Factor'] = 0
df_2['unbalance_Factor'] = 45.9
df_3['unbalance_Factor'] = 60.7
df_4['unbalance_Factor'] = 75.5
df_5['unbalance_Factor'] = 152.1

df_6['unbalance_Factor'] = 0
df_7['unbalance_Factor'] = 45.9
df_8['unbalance_Factor'] = 60.7
df_9['unbalance_Factor'] = 75.5
df_10['unbalance_Factor'] = 152.1

# Training data
# Set sample rate
sampling_rate = 4096
time_interval = 1 / sampling_rate

# Set the start time for data collection
start_time = pd.to_datetime('2024-10-07 00:00:00')

for i in range(1, 6):
    
    df = df_list_train[i-1]
    
    # Add index number to the beginning of column names
    df.columns = [f'{i-1}_{col}' for col in df.columns]
    
    # Add time values to each DataFrame
    df[f'time'] = [start_time + pd.Timedelta(seconds=j * time_interval) for j in range(len(df))]
    
    # Remove the time column using pop and store it in a variable
    time_col = df.pop(f'time')
    
    # Insert the time column at the front
    df.insert(0, f'time', time_col)
    
    # Save the modified DataFrame back to df_list_train
    df_list_train[i-1] = df
    
# Apply transformation to all DataFrames in df_list_train
for i in range(1, 6):
    
    # Transform the data
    df = df_list_train[i-1].melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert the time column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the order of data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert reference time (epoch time) -> UTC time
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

    # Save the transformed DataFrame back to the list
    df_list_train[i-1] = df

# Merge the training data
merged_df_train = pd.concat(df_list_train, axis=0, ignore_index=True)

# Save the training data
merged_df_train.to_csv('./total_train.csv', index=False)

# Test data
# Set sample rate
sampling_rate = 4096
time_interval = 1 / sampling_rate

# Set the start time for data collection
start_time = pd.to_datetime('2024-10-07 00:00:00')

for i in range(1, 6):
    
    df = df_list_test[i-1]
    
    # Add index number to the beginning of column names
    df.columns = [f'{i+4}_{col}' for col in df.columns]
    
    # Add time values to each DataFrame
    df[f'time'] = [start_time + pd.Timedelta(seconds=j * time_interval) for j in range(len(df))]
    
    # Remove the time column using pop and store it in a variable
    time_col = df.pop(f'time')
    
    # Insert the time column at the front
    df.insert(0, f'time', time_col)
    
    # Save the modified DataFrame back to df_list_test
    df_list_test[i-1] = df
    
# Apply transformation to all DataFrames in df_list_test
for i in range(1, 6):
    
    # Transform the data
    df = df_list_test[i-1].melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert the time column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the order of data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert reference time (epoch time) -> UTC time
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))

    # Save the transformed DataFrame back to the list
    df_list_test[i-1] = df
    
# Merge the test data
merged_df_test = pd.concat(df_list_test, axis=0, ignore_index=True)

# Save the test data
merged_df_test.to_csv('./total_test.csv', index=False)

# Create combined DataFrame
merged_df_total = pd.concat([merged_df_train, merged_df_test], axis=0, ignore_index=True)

# Save the combined DataFrame
merged_df_total.to_csv('./datahub-2024-08-vibration_unbalance.csv', index=False) 