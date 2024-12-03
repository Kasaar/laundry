import kagglehub
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from forex_python.converter import CurrencyRates

# Currency conversion function
c = CurrencyRates()
def convert_currency(amount, currency, target_currency="USD"):
    if currency == target_currency:
        return amount
    else:
        rate = c.get_rate(currency, target_currency)
        return amount * rate

class HI_Small_Trans:
    def __init__(self):
        # Download latest version of the dataset
        path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
        file_path = f"{path}/HI-Small_Trans.csv"

        self.df = pd.read_csv(file_path)

        print("Successfully read file")

    def process(self, make_local_copy : bool):
        # Instantiate scaling classes
        scaler = StandardScaler()
        label_encoder = LabelEncoder()

        # Standardize the Timestamp feature
        if 'Timestamp' in self.df.columns:
            # Define the format the timestamp strings are in
            date_format = "%Y/%m/%d %H:%M"

            # Convert each timestamp
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format=date_format).astype(int) // 10**9
            
            self.df['Timestamp'] = scaler.fit_transform(self.df[['Timestamp']])
        else:
            print("Failed to find Timestamp column")

        # Standardize the From Bank feature
        if 'From Bank' in self.df.columns:
            self.df['From Bank'] = scaler.fit_transform(self.df[['From Bank']])
        else:
            print("Failed to find From Bank column")

        # Label encode the Account feature
        if 'Account' in self.df.columns:
            self.df['Account'] = label_encoder.fit_transform(self.df[['Account']])
        else:
            print("Failed to find Account column")

        # Standardize the To Bank feature
        if 'To Bank' in self.df.columns:
            self.df['To Bank'] = scaler.fit_transform(self.df[['To Bank']])
        else:
            print("Failed to find To Bank feature")\

        #  Label encode the Account.1 feature
        if 'Account.1' in self.df.columns:
            self.df['Account.1'] = label_encoder.fit_transform(self.df[['Account.1']])
        else:
            print("Failed to find Account.1 feature")

        # Standardize the Amount Received feature
        if 'Amount Received' in self.df.columns:
            self.df["Amount Received"] = self.df.apply(
                lambda x: convert_currency(x["Amount Received"], x["Receiving Currency"]), axis=1
            )
            self.df['Amount Received'] = scaler.fit_transform(self.df[['Amount Received']])
        else:
            print("Failed to find Amount Received feature")

        # Label encode the Receiving Currency feature
        if 'Receiving Currency' in self.df.columns:
            self.df['Receiving Currency'] = label_encoder.fit_transform(self.df[['Receiving Currency']])
        else:
            print("Failed to find Receiving Currency feature")

        # Standardize the Amount Paid feature
        if 'Amount Paid' in self.df.columns:
            self.df["Amount Paid"] = self.df.apply(
                lambda x: convert_currency(x["Amount Paid"], x["Payment Currency"]), axis=1
            )
            self.df['Amount Paid'] = scaler.fit_transform(self.df[['Amount Paid']])
        else:
            print("Failed to find Amount Paid feature")

        # Label encode the Payment Currency feature
        if 'Payment Currency' in self.df.columns:
            self.df['Payment Currency'] = label_encoder.fit_transform(self.df[['Payment Currency']])
        else:
            print("Failed to find Payment Currency feature")

         # Label encode the Payment Format feature
        if 'Payment Format' in self.df.columns:
            self.df['Payment Format'] = label_encoder.fit_transform(self.df[['Payment Format']])
        else:
            print("Failed to find Payment Format feature")

        if make_local_copy:
            # Store the processed data locally
            file_path = "processed_HI-Small_Trans.csv"
            self.df.to_csv(file_path, index=False)
        
        y = self.df["Is Laundering"]
        self.df.drop(columns=["Receiving Currency", "Payment Currency"], inplace=True)
        X = self.df

        return X, y

if __name__ == "main":
    processer = HI_Small_Trans()
    X, y = processer.process(True)