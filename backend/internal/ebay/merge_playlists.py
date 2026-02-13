import pandas as pd
import glob
import os

# Load all listing CSVs
all_listings = []
print("Current directory:", os.getcwd())
print("\nLooking for CSVs in: backend/internal/ebay/")
for csv_file in glob.glob("backend/internal/ebay/buyitnow_test*.csv"):
    df = pd.read_csv(csv_file)
    print(df)
    all_listings.append(df)

all_listings_df = pd.concat(all_listings, ignore_index=True)
all_listings_df = all_listings_df.drop_duplicates(subset=['Title'])  # Remove duplicates
print(f"Total listings: {len(all_listings_df)}")

# Load keepers CSV
keepers_df = pd.read_csv("backend/internal/ebay/buyitnow_keepers.csv")
keepers_df = keepers_df[keepers_df['Title', 'Price']]
print(f"Total keepers: {len(keepers_df)}")
print(f"Keepers columns: {keepers_df.columns.tolist()}")

# Mark keepers
all_listings_df['wanted'] = all_listings_df['Title'].isin(keepers_df['Title'])
print(len(all_listings_df))

all_listings_df.to_csv('data/ebay_training_data.csv', index=False)

print(f"\nResults:")
print(f"Keepers: {all_listings_df['wanted'].sum()}")
print(f"Skips: {(~all_listings_df['wanted']).sum()}")
print(f"Keeper rate: {all_listings_df['wanted'].mean()*100:.2f}%")
