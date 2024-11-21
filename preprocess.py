import kagglehub
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Download latest version of the dataset
path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
file_path = f"{path}/HI-Small_Trans.csv"
df = pd.read_csv(file_path)

print("Successfully read file")

# Convert timestamps to UNIX format
if 'Timestamp' in df.columns:
    # Define the format the timestamp strings are in
    date_format = "%Y/%m/%d %H:%M"

    # Convert each timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=date_format).astype(int) // 10**9
    
    # Standardize timestamp feature
    scaler = StandardScaler()
    df['Timestamp_Scaled'] = scaler.fit_transform(df[['Timestamp']])

# Standardize sending bank designation

# Standardize sending account number

# Standardize receiving bank designation

#  Standardize receiving account number
