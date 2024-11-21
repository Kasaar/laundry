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
    
    # Standardize the timestamp feature
    scaler = StandardScaler()
    df['Timestamp_Scaled'] = scaler.fit_transform(df[['Timestamp']])

    # Replace the 'Timestamp' column with the scaled values
    df['Timestamp'] = df['Timestamp_Scaled']
    df.drop(columns=['Timestamp_Scaled'], inplace=True)

# Standardize sending bank designation

# Standardize sending account number

# Standardize receiving bank designation

#  Standardize receiving account number

# Store the processed data locally
output_file_path = "processed_HI-Small_Trans.csv"
df.to_csv(output_file_path, index=False)