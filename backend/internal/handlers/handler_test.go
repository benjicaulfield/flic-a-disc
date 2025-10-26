package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

	"flic-a-disc/internal/config"
	"flic-a-disc/internal/models"
)

func setupTestDB(t *testing.T) *gorm.DB {
	// Create in-memory SQLite database for testing
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to connect to test database: %v", err)
	}

	// Migrate the schema
	err = db.AutoMigrate(&models.Record{}, &models.DiscogsListing{})
	if err != nil {
		t.Fatalf("Failed to migrate test database: %v", err)
	}

	return db
}

func TestLabelRecords(t *testing.T) {
	// Setup
	db := setupTestDB(t)
	cfg := &config.Config{}
	handler := New(db, cfg)

	// Create test data
	record := models.Record{
		DiscogsID: "123456",
		Artist:    "Test Artist",
		Title:     "Test Album",
		Evaluated: false,
		Wanted:    false,
	}
	db.Create(&record)

	listing := models.DiscogsListing{
		RecordID:       record.ID,
		RecordPrice:    "10.00",
		MediaCondition: "Very Good",
	}
	db.Create(&listing)

	// Create request payload
	requestBody := LabelRequest{
		Labels: []struct {
			ID    uint `json:"id"`
			Label bool `json:"label"`
		}{
			{ID: listing.ID, Label: true},
		},
	}

	jsonBody, _ := json.Marshal(requestBody)

	// Setup Gin test context
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/discogs/labels", bytes.NewBuffer(jsonBody))
	c.Request.Header.Set("Content-Type", "application/json")

	// Execute
	handler.LabelRecords(c)

	// Assert HTTP response
	assert.Equal(t, http.StatusOK, w.Code)

	// Assert database was updated correctly
	var updatedRecord models.Record
	db.First(&updatedRecord, record.ID)

	assert.True(t, updatedRecord.Evaluated, "Record should be marked as evaluated")
	assert.True(t, updatedRecord.Wanted, "Record should be marked as wanted")
}

func TestLabelRecords_MultipleLabels(t *testing.T) {
	// Setup
	db := setupTestDB(t)
	cfg := &config.Config{}
	handler := New(db, cfg)

	// Create multiple test records
	record1 := models.Record{DiscogsID: "111", Artist: "Artist 1", Evaluated: false, Wanted: false}
	record2 := models.Record{DiscogsID: "222", Artist: "Artist 2", Evaluated: false, Wanted: false}
	db.Create(&record1)
	db.Create(&record2)

	listing1 := models.DiscogsListing{RecordID: record1.ID, RecordPrice: "10.00"}
	listing2 := models.DiscogsListing{RecordID: record2.ID, RecordPrice: "15.00"}
	db.Create(&listing1)
	db.Create(&listing2)

	// Label one as wanted, one as not wanted
	requestBody := LabelRequest{
		Labels: []struct {
			ID    uint `json:"id"`
			Label bool `json:"label"`
		}{
			{ID: listing1.ID, Label: true},
			{ID: listing2.ID, Label: false},
		},
	}

	jsonBody, _ := json.Marshal(requestBody)

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/discogs/labels", bytes.NewBuffer(jsonBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler.LabelRecords(c)

	// Assert
	var updatedRecord1, updatedRecord2 models.Record
	db.First(&updatedRecord1, record1.ID)
	db.First(&updatedRecord2, record2.ID)

	assert.True(t, updatedRecord1.Wanted, "Record 1 should be wanted")
	assert.False(t, updatedRecord2.Wanted, "Record 2 should not be wanted")
	assert.True(t, updatedRecord1.Evaluated, "Record 1 should be evaluated")
	assert.True(t, updatedRecord2.Evaluated, "Record 2 should be evaluated")
}
