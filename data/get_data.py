import kagglehub

# Download latest version
path = kagglehub.dataset_download("siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data")

print("Path to dataset files:", path)