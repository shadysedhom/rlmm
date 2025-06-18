import pandas as pd
PATH = "data/raw/kaggle/btcusdt-perps-limit-order-book-data-1-09-1-20.csv"

# -- Looking at the data

# ❶ Read only the first few lines, *without* assuming a header row
peek = pd.read_csv(PATH, nrows=5, header=None)
print("shape:", peek.shape)      # how many columns pandas sees
print(peek.head().to_string())   # raw values

peek_last = pd.read_csv(PATH, nrows=1000, header=None, skiprows=1)
print("non-null values in last col:", peek_last.iloc[:, -1].notna().sum())