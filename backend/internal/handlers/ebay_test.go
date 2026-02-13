package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/benjicaulfield/flic-a-disc/internal/ebay"
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/stretchr/testify/assert"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func init() {
	// Load .env file if it exists (ignoring errors if it doesn't)
	_ = godotenv.Load("../../.env.local")
}

// setupTestDB creates an in-memory SQLite database for testing
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

// setupTestHandlerNoClient creates handler without eBay client for unit tests
func setupTestHandlerNoClient(t *testing.T) *EbayHandler {
	db := setupTestDB(t)
	return &EbayHandler{
		db: db,
	}
}

// setupTestHandler creates a handler with real eBay client for integration tests
func setupTestHandler(t *testing.T) *EbayHandler {
	db := setupTestDB(t)

	appID := os.Getenv("EBAY_APP_ID")
	certID := os.Getenv("EBAY_CERT_ID")

	if appID == "" || certID == "" {
		t.Skip("EBAY_APP_ID and EBAY_CERT_ID must be set for integration tests")
	}

	return NewEbayHandler(appID, certID, db)
}

// ========== UNIT TESTS (don't need eBay API) ==========

func TestGetCachedListings_Empty(t *testing.T) {
	handler := setupTestHandlerNoClient(t)

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.GetCachedListings(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)

	listings := response["listings"].([]interface{})
	assert.Equal(t, 0, len(listings))
}

func TestGetCachedListings_WithData(t *testing.T) {
	handler := setupTestHandlerNoClient(t)

	handler.cachedListings = []gin.H{
		{
			"ebay_id":     "123456",
			"ebay_title":  "Test Record",
			"price":       "25.00",
			"current_bid": "30.00",
		},
		{
			"ebay_id":     "789012",
			"ebay_title":  "Another Record",
			"price":       "15.00",
			"current_bid": "18.00",
		},
	}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.GetCachedListings(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)

	listings := response["listings"].([]interface{})
	assert.Equal(t, 2, len(listings))

	firstListing := listings[0].(map[string]interface{})
	assert.Equal(t, "123456", firstListing["ebay_id"])
	assert.Equal(t, "Test Record", firstListing["ebay_title"])
}

func TestTriggerFetchAuctions_AlreadyInProgress(t *testing.T) {
	handler := setupTestHandlerNoClient(t)
	handler.fetchInProgress = true

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.TriggerFetchAuctions(c)

	assert.Equal(t, http.StatusConflict, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "Fetch already in progress", response["message"])
}

func TestGetUnannotatedListings_EmptyCache(t *testing.T) {
	handler := setupTestHandlerNoClient(t)

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.GetUnannotatedListings(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)

	assert.Equal(t, float64(0), response["total"])
	assert.Equal(t, "No listings cached. Click Refresh to fetch from eBay.", response["message"])
}

func TestGetUnannotatedListings_WithCache(t *testing.T) {
	handler := setupTestHandlerNoClient(t)

	handler.cachedListings = []gin.H{
		{"ebay_id": "111", "ebay_title": "Record 1"},
		{"ebay_id": "222", "ebay_title": "Record 2"},
		{"ebay_id": "333", "ebay_title": "Record 3"},
	}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.GetUnannotatedListings(c)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)

	assert.Equal(t, float64(3), response["total"])
	listings := response["listings"].([]interface{})
	assert.Equal(t, 3, len(listings))
}

func TestParseRecordMetadata(t *testing.T) {
	item := &ebay.ItemSummary{
		LocalizedAspects: []ebay.LocalizedAspect{
			{Name: "Artist", Value: "John Coltrane"},
			{Name: "Release Title", Value: "A Love Supreme"},
			{Name: "Record Label", Value: "Impulse!"},
			{Name: "Format", Value: "LP"},
			{Name: "Release Year", Value: "1965"},
			{Name: "Record Grading", Value: "NM"},
			{Name: "Sleeve Grading", Value: "VG+"},
			{Name: "Genre", Value: "Jazz"},
			{Name: "Style", Value: "Spiritual Jazz"},
		},
	}

	artist, album, label, format, year, recordCondition, sleeveCondition, genre, style := parseRecordMetadata(item)

	assert.Equal(t, "John Coltrane", artist)
	assert.Equal(t, "A Love Supreme", album)
	assert.Equal(t, "Impulse!", label)
	assert.Equal(t, "LP", format)
	assert.Equal(t, "1965", year)
	assert.Equal(t, "NM", recordCondition)
	assert.Equal(t, "VG+", sleeveCondition)
	assert.Equal(t, "Jazz", genre)
	assert.Equal(t, "Spiritual Jazz", style)
}

func TestParseRecordMetadata_AlternativeFieldNames(t *testing.T) {
	item := &ebay.ItemSummary{
		LocalizedAspects: []ebay.LocalizedAspect{
			{Name: "Album", Value: "Blue Train"},
			{Name: "Label", Value: "Blue Note"},
			{Name: "Year", Value: "1957"},
		},
	}

	_, album, label, _, year, _, _, _, _ := parseRecordMetadata(item)

	assert.Equal(t, "Blue Train", album)
	assert.Equal(t, "Blue Note", label)
	assert.Equal(t, "1957", year)
}

func TestConcurrentAccess(t *testing.T) {
	handler := setupTestHandlerNoClient(t)

	handler.cachedListings = []gin.H{
		{"ebay_id": "initial", "ebay_title": "Initial Listing"},
	}

	gin.SetMode(gin.TestMode)
	done := make(chan bool)

	// Concurrent readers
	for i := 0; i < 10; i++ {
		go func() {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			handler.GetCachedListings(c)
			assert.Equal(t, http.StatusOK, w.Code)
			done <- true
		}()
	}

	// Concurrent cache updates
	for i := 0; i < 5; i++ {
		go func(id int) {
			handler.mu.Lock()
			handler.cachedListings = append(handler.cachedListings, gin.H{
				"ebay_id":    "concurrent_" + string(rune(id)),
				"ebay_title": "Concurrent Listing",
			})
			handler.mu.Unlock()
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 15; i++ {
		<-done
	}

	// Verify no race conditions
	handler.mu.RLock()
	cacheLen := len(handler.cachedListings)
	handler.mu.RUnlock()
	assert.True(t, cacheLen >= 1 && cacheLen <= 6)
}

// ========== INTEGRATION TESTS (need eBay API) ==========

func TestSaveSelectedListings_Integration(t *testing.T) {
	handler := setupTestHandler(t) // Will skip if no API keys

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	// Use a real eBay item ID (you'll need to find a current auction)
	// This test will make real API calls
	requestBody := map[string]interface{}{
		"ebay_ids": []string{"166029511878"}, // Example ID - replace with active auction
	}
	jsonBody, _ := json.Marshal(requestBody)
	c.Request = httptest.NewRequest("POST", "/api/ebay/save", bytes.NewBuffer(jsonBody))
	c.Request.Header.Set("Content-Type", "application/json")

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)

	// Check that listing was saved to database
	var count int64
	handler.db.Model(&models.EbayListing{}).Count(&count)
	assert.Greater(t, count, int64(0))
}

// Benchmark tests
func BenchmarkGetCachedListings(b *testing.B) {
	db, _ := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	handler := &EbayHandler{db: db}

	// Populate with realistic data
	for i := 0; i < 1000; i++ {
		handler.cachedListings = append(handler.cachedListings, gin.H{
			"ebay_id":     "bench_" + string(rune(i)),
			"ebay_title":  "Benchmark Listing",
			"price":       "25.00",
			"current_bid": "30.00",
		})
	}

	gin.SetMode(gin.TestMode)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		handler.GetCachedListings(c)
	}
}
