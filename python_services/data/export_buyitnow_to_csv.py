import csv

def export_buyitnow_to_csv():
    from ..bandit.enhance_listings import LookupByID
    
    
    items = client.search_buy_it_now()
    
    with open('buyitnow_test44.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Price', 'EbayID'])
        
        for item in items:
            price = item.get('price', {}).get('value', '')
            ebay_id = item.get('itemId', '')
            title = item.get('title', '')
            
            writer.writerow([title, price, ebay_id])
    
    print(f"Wrote {len(items)} items to buyitnow_test44.csv")

if __name__ == '__main__':
    export_buyitnow_to_csv()