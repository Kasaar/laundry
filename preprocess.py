import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")

# Convert timestamps to UNIX format

# Standardize sending bank designation

# Standardize sending account number

# Standardize receiving bank designation

#  Standardize receiving account number

print("Path to dataset files:", path)