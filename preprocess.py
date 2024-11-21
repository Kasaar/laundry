import kagglehub
import pandas as pd
from datetime import datetime

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
    for ele in df['Timestamp']:
        # Convert to datetime object
        date_time = datetime.strptime(ele, date_format)

        # Convert to UNIX timestamp
        unix_timestamp = int(date_time.timestamp())

        print(unix_timestamp)

# Standardize sending bank designation

# Standardize sending account number

# Standardize receiving bank designation

#  Standardize receiving account number
