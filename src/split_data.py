# Cleaning the .csv files, removing unnecessary fields.
# Splitting the data into training and testing data and outputting them as .csv files.
import pandas as pd
from sklearn.model_selection import train_test_split


# Converting the .csv file to a pandas DataFrame.
raw_data = pd.read_csv("data/raw_data.csv")
# Removing unnecessary, sensitive columns.
clean_data = raw_data.drop(columns=["Transaction Date", "Transaction Type", "Sort Code", "Account Number", "Balance"])
# Replacing all NaNs with zeros to allow for an 'amount' to be calculated and associated to each transaction.
clean_data["Credit Amount"] = clean_data["Credit Amount"].fillna(0)
clean_data["Debit Amount"] = clean_data["Debit Amount"].fillna(0)
clean_data["Amount"] = clean_data["Credit Amount"] - clean_data["Debit Amount"]
# Dropping the credit and amount fields.
clean_data = clean_data.drop(columns=["Credit Amount", "Debit Amount"])


# Splitting this into a dataframe for training the model and for testing the model.
# Using 20% of the data for testing.
# Using stratify to ensure that there are a proportionate number of categories in both dataframes.
train_data, test_data = train_test_split(clean_data, test_size=0.2, stratify=clean_data["Category"])


# Saving the DataFrames as separate .csv files.
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)
