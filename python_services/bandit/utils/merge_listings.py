import pandas as pd
import glob

# Use absolute paths
all_listings = []
for csv_file in glob.glob("/Users/benjamincaulfield/Documents/flic-a-disc/backend/internal/ebay/buyitnow_test*.csv"):
    df = pd.read_csv(csv_file)
    all_listings.append(df)

if len(all_listings) == 0:
    print("No CSV files found!")
    exit()

all_listings_df = pd.concat(all_listings, ignore_index=True)
all_listings_df = all_listings_df.drop_duplicates(subset=['Title'])
print(f"Total listings: {len(all_listings_df)}")

# Load keepers
keepers_df = pd.read_csv("/Users/benjamincaulfield/Documents/flic-a-disc/backend/internal/ebay/buyitnow_keepers.csv")
print(f"Total keepers: {len(keepers_df)}")
print(f"Keepers columns: {keepers_df.columns.tolist()}")

# Mark keepers
all_listings_df['wanted'] = all_listings_df['Title'].isin(keepers_df['Title'])

# Save
all_listings_df.to_csv('/Users/benjamincaulfield/Documents/flic-a-disc/python_services/data/ebay_training_data.csv', index=False)

print(f"\nResults:")
print(f"Keepers: {all_listings_df['wanted'].sum()}")
print(f"Skips: {(~all_listings_df['wanted']).sum()}")
print(f"Keeper rate: {all_listings_df['wanted'].mean()*100:.2f}%")