package ebay

import (
	"encoding/csv"
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/joho/godotenv"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func init() {
	err := godotenv.Load("../../.env.local")
	if err != nil {
		log.Printf("Error loading .env: %v", err)
	} else {
		log.Println(".env loaded successfully")
	}
}

func setupTestDB(t *testing.T) *gorm.DB {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to connect to test database: %v", err)
	}

	err = db.AutoMigrate(&models.EbayListing{})
	if err != nil {
		t.Fatalf("Failed to migrate test database: %v", err)
	}

	return db
}

func getClient(t *testing.T) (*Client, error) {
	appID := os.Getenv("EBAY_APP_ID")
	certID := os.Getenv("EBAY_CERT_ID")
	if appID == "" || certID == "" {
		t.Skip("EBAY_APP_ID and EBAY_CERT_ID must be set for integration tests")
	}
	client := NewClient(appID, certID)
	err := client.GetAccessToken()
	return client, err
}

func TestGetClient(t *testing.T) {
	client, err := getClient(t)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestBuyItNowToCSV(t *testing.T) {
	client, err := getClient(t)
	assert.NoError(t, err)

	items, err := client.SearchBuyItNow()
	if err != nil {
		t.Fatal(err)
	}

	file, _ := os.Create("buyitnow_test44.csv")
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"EbayID", "Title", "Price"})

	for _, item := range items {
		price := ""
		if item.Price.Value != "" {
			price = item.Price.Value
		}

		writer.Write([]string{
			item.ItemID,
			item.Title,
			price,
		})
	}

	log.Printf("Wrote %d items to buyitnow_test.csv", len(items))
}
