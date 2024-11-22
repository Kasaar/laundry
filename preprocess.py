import kagglehub
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Download latest version of the dataset
path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
file_path = f"{path}/HI-Small_Trans.csv"
df = pd.read_csv(file_path)

print("Successfully read file")

# Convert timestamps to UNIX format and standardize
if 'Timestamp' in df.columns:
    # Define the format the timestamp strings are in
    date_format = "%Y/%m/%d %H:%M"

    # Convert each timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=date_format).astype(int) // 10**9
    
    # Standardize the Timestamp feature
    scaler = StandardScaler()
    df['Timestamp'] = scaler.fit_transform(df[['Timestamp']])
else:
    print("Failed to find timestamp column")

# Standardize sending bank designation
if 'From Bank' in df.columns:
    # Standardize the From Bank feature
    scaler = StandardScaler()
    df['From Bank'] = scaler.fit_transform(df[['From Bank']])
else:
    print("Failed to find From Bank column")

# Standardize sending account number

# Standardize receiving bank designation

#  Standardize receiving account number

# Store the processed data locally
output_file_path = "processed_HI-Small_Trans.csv"
df.to_csv(output_file_path, index=False)