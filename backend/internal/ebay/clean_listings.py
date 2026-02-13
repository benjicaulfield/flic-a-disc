# clean_listings.py
import csv

input_file = 'listings.csv'
output_file = 'listings_cleaned.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    for row in reader:
        # Keep only first 2 fields (title, price)
        writer.writerow(row[:2])

print(f"Cleaned {input_file} -> {output_file}")