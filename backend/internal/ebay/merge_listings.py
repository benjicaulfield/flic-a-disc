import pandas as pd
import glob
import os

# Load all buyitnow CSVs FROM CURRENT DIRECTORY
all_listings = []
print("Current directory:", os.getcwd())

for csv_file in glob.glob("buyitnow_test*.csv"):  # No path prefix needed!
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    all_listings.append(df)

if not all_listings:
    print("No CSV files found!")
else:
    all_listings_df = pd.concat(all_listings, ignore_index=True)
    all_listings_df = all_listings_df.drop_duplicates(subset=['Title'])
    print(f"\nTotal listings: {len(all_listings_df)}")

    all_listings_df.to_csv('ebay_training_data.csv', index=False)

    print(f"\nResults:")
    