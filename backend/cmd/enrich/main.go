package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/joho/godotenv"

	"github.com/benjicaulfield/flic-a-disc/internal/ebay"
)

func init() {
	err := godotenv.Load("../../.env.local")
	if err != nil {
		log.Printf("Error loading .env: %v", err)
	} else {
		log.Println(".env loaded successfully")
	}
}

func main() {
	// Initialize eBay client
	appID := os.Getenv("EBAY_APP_ID")
	certID := os.Getenv("EBAY_CERT_ID")

	client := ebay.NewClient(appID, certID)
	if err := client.GetAccessToken(); err != nil {
		log.Fatal(err)
	}

	// Read input CSV
	inputFile, err := os.Open("listings.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer inputFile.Close()

	reader := csv.NewReader(inputFile)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Create output CSV
	outputFile, err := os.Create("enhanced_listings.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	// Write header
	writer.Write([]string{
		"EbayID", "Title", "Price", "Artist", "Album", "Label",
		"Year", "Genre", "Style", "RecordCondition", "SleeveCondition", "Format",
	})

	// Process each record (skip header if exists)
	for i, record := range records {
		if i == 0 && record[0] == "EbayID" {
			continue
		}

		// Extract eBay ID from first column (format: v1|ID|0)

		ebayID := record[0]
		title := record[1]
		price := record[2]

		// Fetch item details
		item, err := client.LookupByItemID(ebayID)
		if err != nil {
			log.Printf("Error fetching item %s: %v", ebayID, err)
			writer.Write([]string{ebayID, title, price, "", "", "", "", "", "", "", "", ""})
			continue
		}

		// Parse metadata
		artist, album, label, format, year, recordCond, sleeveCond, genre, style := parseRecordMetadata(item)

		writer.Write([]string{
			ebayID, title, price, artist, album, label,
			year, genre, style, recordCond, sleeveCond, format,
		})

		log.Printf("Processed %d/%d: %s", i+1, len(records), ebayID)
		time.Sleep(200 * time.Millisecond)
	}

	fmt.Printf("Enhanced %d listings -> buyitnow_enhanced.csv\n", len(records)-1)
}

func parseRecordMetadata(item *ebay.ItemSummary) (artist, album, label, format, year, recordCondition, sleeveCondition, genre, style string) {
	for _, aspect := range item.LocalizedAspects {
		switch aspect.Name {
		case "Artist":
			artist = aspect.Value
		case "Release Title", "Album":
			if album == "" {
				album = aspect.Value
			}
		case "Record Label", "Label":
			if label == "" {
				label = aspect.Value
			}
		case "Format", "Record Size":
			if format == "" {
				format = aspect.Value
			}
		case "Release Year", "Year":
			if year == "" {
				year = aspect.Value
			}
		case "Record Grading":
			recordCondition = aspect.Value
		case "Sleeve Grading":
			sleeveCondition = aspect.Value
		case "Genre":
			genre = aspect.Value
		case "Style":
			style = aspect.Value
		}
	}
	return
}
