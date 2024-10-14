# Import necessary libraries
import pandas as pd 

# Load MIT-BIH data
df_mit_bih_train = pd.read_csv('./mitbih_train.csv')
df_mit_bih_test = pd.read_csv('./mitbih_test.csv')

# Load PTB data
df_ptb_normal = pd.read_csv('./ptbdb_normal.csv')
df_ptb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')

# Rename columns
# Change the first to the 187th columns to data_name_train_number and data_name_test_number
# Change the last column to data_name_train_label

# MIT-BIH data
df_mit_bih_train.columns = [f"mit_bih_train_{i}" for i in range(187)] + ['mit_bih_train_label']
df_mit_bih_test.columns = [f"mit_bih_test_{i}" for i in range(187)] + ['mit_bih_test_label']

# PTB data
df_ptb_normal.columns = [f"ptb_{i}" for i in range(187)] + ['ptb_label']
df_ptb_abnormal.columns = [f"ptb_{i}" for i in range(187)] + ['ptb_label']

# Combine PTB data
df_ptb = pd.concat([df_ptb_normal, df_ptb_abnormal], axis=0, ignore_index=True)

# Convert label column data to integer type
df_mit_bih_train['mit_bih_train_label'] = df_mit_bih_train['mit_bih_train_label'].astype(int)
df_mit_bih_test['mit_bih_test_label'] = df_mit_bih_test['mit_bih_test_label'].astype(int)
df_ptb['ptb_label'] = df_ptb['ptb_label'].astype(int)

# Shuffle the dataframe since labels are in order
df_mit_bih_train = df_mit_bih_train.sample(frac=1, random_state=77).reset_index(drop=True)
df_mit_bih_test = df_mit_bih_test.sample(frac=1, random_state=77).reset_index(drop=True)
df_ptb = df_ptb.sample(frac=1, random_state=77).reset_index(drop=True)

# Generate time
df_mit_bih_time = pd.DataFrame()
df_mit_bih_time['time'] = pd.date_range(start='2024-10-14 00:00:00', periods=(len(df_mit_bih_train) + len(df_mit_bih_test)), freq='T')

df_ptb_time = pd.DataFrame()
df_ptb_time['time'] = pd.date_range(start='2024-10-14 00:00:00', periods=len(df_ptb), freq='T')

# Set time
df_mit_bih_train['time'] = df_mit_bih_time['time'].values[:len(df_mit_bih_train)]
df_mit_bih_test['time'] = df_mit_bih_time['time'].values[len(df_mit_bih_train):]

df_ptb['time'] = df_ptb_time.values

# Move the last column to the front
last_column_train = df_mit_bih_train.columns[-1] 
df_mit_bih_train.insert(0, last_column_train, df_mit_bih_train.pop(last_column_train))

last_column_test = df_mit_bih_test.columns[-1] 
df_mit_bih_test.insert(0, last_column_test, df_mit_bih_test.pop(last_column_test))

last_column_ptb = df_ptb.columns[-1] 
df_ptb.insert(0, last_column_ptb, df_ptb.pop(last_column_ptb))

# Data transformation function
def data_change_db(df):
    
    # Transform data
    df = df.melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert time column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the order of data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert epoch time (UTC time)
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Data transformation
df_mit_bih_train = data_change_db(df_mit_bih_train)
df_mit_bih_test = data_change_db(df_mit_bih_test)
df_ptb = data_change_db(df_ptb)

# Merge data
df_total = pd.concat([df_mit_bih_train, df_mit_bih_test, df_ptb], axis=0, ignore_index=True)

# Save the combined dataframe
df_total.to_csv('./datahub-2024-10-ECG-HeartBeat.csv', index=False)