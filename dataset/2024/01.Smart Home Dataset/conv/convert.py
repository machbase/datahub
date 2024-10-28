# Import necessary libraries
import pandas as pd 

# Load data
df = pd.read_csv('./HomeC.csv', low_memory=False)

# Remove missing values
df = df.dropna()

# Remove unnecessary columns 
df = df.drop(['icon', 'summary', 'cloudCover'], axis=1)

# Transform data
df = df.melt(id_vars=['time'], var_name='name', value_name='value')

# Save DataFrame
df.to_csv('./datahub-2024-1-home.csv', index=False)