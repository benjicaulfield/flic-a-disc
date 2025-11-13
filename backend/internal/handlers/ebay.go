package handlers

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/benjicaulfield/flic-a-disc/internal/ebay"
	"github.com/benjicaulfield/flic-a-disc/internal/models"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

type EbayHandler struct {
	ebayClient *ebay.Client
	db         *gorm.DB
	mu         sync.RWMutex
	listings   []gin.H
}

func NewEbayHandler(appID, certID string, db *gorm.DB) *EbayHandler {
	h := &EbayHandler{
		ebayClient: ebay.NewClient(appID, certID),
		db:         db,
	}
	return h
}

func (h *EbayHandler) fetchAndCacheListings() {
	log.Println("Fetching eBay auctions ending in next 24-48 hours...")

	allResults, err := h.ebayClient.SearchAuctionsEndingSoon(24)
	if err != nil {
		log.Printf("Failed to fetch listings: %v", err)
		return
	}

	now := time.Now()
	cutoffEnd := now.Add(24 * time.Hour)

	var raw []gin.H
	for _, item := range allResults {
		endDate, err := time.Parse(time.RFC3339, item.ItemEndDate)
		if err != nil {
			continue
		}

		if endDate.Before(cutoffEnd) && endDate.After(now) {
			var priceValue string
			if item.Price != nil {
				priceValue = item.Price.Value
			}

			currentBid := priceValue
			if item.CurrentBidPrice != nil {
				currentBid = item.CurrentBidPrice.Value
			}

			raw = append(raw, gin.H{
				"ebay_id":     item.ItemID,
				"title":       item.Title,
				"price":       priceValue,
				"current_bid": currentBid,
				"bid_count":   item.BidCount,
				"url":         item.ItemWebURL,
				"end_date":    item.ItemEndDate,
			})
		}
	}

	log.Printf("Found %d raw listings from ebay", len(raw))

	filteredListings, err := h.filterListingsByTFIDF(raw)
	if err != nil {
		log.Printf("Failed to filter listings: %v", err)
		return
	}

	log.Printf("TF-IDF filter passed %d/%d listings", len(filteredListings), len(raw))

	// ✅ Save only filtered listings to database
	for _, item := range allResults {
		// Check if this item passed the filter
		passed := false
		for _, filtered := range filteredListings {
			if filtered["ebay_id"].(string) == item.ItemID {
				passed = true
				break
			}
		}

		if !passed {
			continue // Skip listings that didn't pass filter
		}

		var priceValue string
		if item.Price != nil {
			priceValue = item.Price.Value
		}

		currentBid := priceValue
		if item.CurrentBidPrice != nil {
			currentBid = item.CurrentBidPrice.Value
		}

		endDate, _ := time.Parse(time.RFC3339, item.ItemEndDate)
		creationDate, _ := time.Parse(time.RFC3339, item.ItemCreationDate)

		dbListing := models.EbayListing{
			EbayID:       item.ItemID,
			EbayTitle:    item.Title,
			Price:        priceValue,
			CurrentBid:   currentBid,
			BidCount:     item.BidCount,
			EndDate:      endDate,
			CreationDate: creationDate,
		}

		if err := h.db.Create(&dbListing).Error; err != nil {
			log.Printf("Failed to save listing %s: %v", item.ItemID, err)
		}
	}

	log.Printf("Saved %d filtered listings to database", len(filteredListings))
}

// ✅ New helper function to call Python filter
func (h *EbayHandler) filterListingsByTFIDF(listings []gin.H) ([]gin.H, error) {
	filterURL := "http://localhost:8001/ml/ebay_title_similarity_filter/"

	requestBody := map[string]interface{}{
		"listings": listings,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(filterURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Passed []gin.H `json:"passed"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Passed, nil
}

func (h *EbayHandler) TriggerFetch(c *gin.Context) {
	h.fetchAndCacheListings()
	c.JSON(200, gin.H{"message": "Fetch complete"})
}

func (h *EbayHandler) GetEbayAuctionsPage(c *gin.Context) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	c.JSON(200, gin.H{
		"listings": h.listings,
		"count":    len(h.listings),
	})
}

func (h *EbayHandler) saveToCSV(listings []gin.H) error {
	filename := time.Now().Format("ebay_auctions_2006-01-02.csv")
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	writer.Write([]string{"ebay_id", "title", "price", "current_bid", "bid_count", "url", "end_date"})

	// Data
	for _, l := range listings {
		writer.Write([]string{
			l["ebay_id"].(string),
			l["title"].(string),
			l["price"].(string),
			l["current_bid"].(string),
			fmt.Sprintf("%d", l["bid_count"].(int)),
			l["url"].(string),
			l["end_date"].(string),
		})
	}

	log.Printf("Saved to %s", filename)
	return nil
}

func (h *EbayHandler) SaveSelectedListings(c *gin.Context) {
	var req struct {
		EbayIDs []string `json:"ebay_ids"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}

	for _, ebayID := range req.EbayIDs {
		item, err := h.ebayClient.LookupByItemID(ebayID)
		if err != nil {
			log.Printf("Failed to lookup %s: %v", ebayID, err)
			continue
		}

		artist, album, label, format, year, recordCondition, sleeveCondition, genre, style := parseRecordMetadata(item)

		endDate, _ := time.Parse(time.RFC3339, item.ItemEndDate)
		creationDate, _ := time.Parse(time.RFC3339, item.ItemCreationDate)

		currentBid := item.Price.Value
		if item.CurrentBidPrice != nil {
			currentBid = item.CurrentBidPrice.Value
		}

		listing := models.EbayListing{
			EbayID:          ebayID,
			EbayTitle:       item.Title,
			Price:           item.Price.Value,
			CurrentBid:      currentBid,
			BidCount:        item.BidCount,
			EndDate:         endDate,
			CreationDate:    creationDate,
			Artist:          artist,
			Album:           album,
			Label:           label,
			Format:          format,
			Year:            year,
			RecordCondition: recordCondition,
			SleeveCondition: sleeveCondition,
			Genre:           genre,
			Style:           style,
			Saved:           true,
		}

		h.db.Create(&listing)
	}

	c.JSON(200, gin.H{"message": "Saved", "count": len(req.EbayIDs)})
}

func (h *EbayHandler) GetSavedListings(c *gin.Context) {
	var listings []models.EbayListing
	h.db.Where("saved = ?", true).Order("created_at DESC").Find(&listings)

	c.JSON(200, gin.H{
		"listings": listings,
		"count":    len(listings),
	})
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

func (h *EbayHandler) RecommendEbayListings(c *gin.Context) {
	log.Println("🚀 Starting eBay recommendation pipeline...")
	log.Println("📊 Stage 1: Getting predictions from ML service...")

	firstPassURL := "http://localhost:8001/ml/ebay_first_pass/"
	requestBody := map[string]int{"top_n": 500}
	jsonBody, _ := json.Marshal(requestBody)

	resp, err := http.Post(firstPassURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		log.Printf("Failed to call ML service: %v", err)
		c.JSON(500, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()

	var firstPassResult struct {
		Candidates []struct {
			EbayID      string  `json:"ebay_id"`
			Prediction  float64 `json:"prediction"`
			Uncertainty float64 `json:"uncertainty"`
			Title       string  `json:"title"`
		} `json:"candidates"`
		TotalProcessed int `json:"total_processed"`
		TopN           int `json:"top_n"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&firstPassResult); err != nil {
		log.Printf("Failed to decode ML response: %v", err)
		c.JSON(500, gin.H{"error": "Invalid ML response"})
		return
	}

	log.Printf("✅ Stage 1 complete: %d candidates from %d total listings",
		firstPassResult.TopN, firstPassResult.TotalProcessed)

	// STAGE 2: Enrich top candidates with eBay API metadata
	log.Println("🔍 Stage 2: Enriching candidates with eBay metadata...")

	enrichedCount := 0
	for i, candidate := range firstPassResult.Candidates {
		if i >= 200 { // Limit API calls to 200
			break
		}

		// Call eBay API to get full metadata
		itemDetails, err := h.ebayClient.LookupByItemID(candidate.EbayID)
		if err != nil {
			log.Printf("⚠️  Failed to lookup %s: %v", candidate.EbayID, err)
			continue
		}

		// Parse metadata from eBay response
		artist, album, label, format, year, recordCondition, sleeveCondition, genre, style :=
			parseRecordMetadata(itemDetails)

		h.db.Model(&models.EbayListing{}).
			Where("ebay_id = ?", candidate.EbayID).
			Updates(map[string]interface{}{
				"artist":           artist,
				"album":            album,
				"label":            label,
				"format":           format,
				"year":             year,
				"record_condition": recordCondition,
				"sleeve_condition": sleeveCondition,
				"genre":            genre,
				"style":            style,
				"metadata_fetched": true,
			})

		enrichedCount++

		if enrichedCount%50 == 0 {
			log.Printf("   Enriched %d/%d...", enrichedCount, 200)
		}

		// Rate limiting - be nice to eBay API
		time.Sleep(100 * time.Millisecond)
	}

	log.Printf("✅ Stage 2 complete: Enriched %d listings with metadata", enrichedCount)

	// STAGE 3: Call ML service again with full features for final ranking
	log.Println("🎯 Stage 3: Final ranking with full features...")

	// Get enriched listings from database
	var enrichedListings []models.EbayListing
	h.db.Where("metadata_fetched = ?", true).
		Order("updated_at DESC").
		Limit(enrichedCount).
		Find(&enrichedListings)

	// TODO: Call ML service with full features
	// For now, just return the enriched listings sorted by Stage 1 score
	log.Printf("🔍 Checking %d enriched listings against %d candidates", len(enrichedListings), len(firstPassResult.Candidates))

	// Map enriched listings back to predictions
	results := []gin.H{}
	for _, listing := range enrichedListings {
		var pred float64
		for _, candidate := range firstPassResult.Candidates {
			if candidate.EbayID == listing.EbayID {
				pred = candidate.Prediction
				break
			}
		}

		results = append(results, gin.H{
			"id":          listing.ID,
			"ebay_id":     listing.EbayID,
			"title":       listing.EbayTitle,
			"artist":      listing.Artist,
			"album":       listing.Album,
			"label":       listing.Label,
			"genre":       listing.Genre,
			"style":       listing.Style,
			"year":        listing.Year,
			"format":      listing.Format,
			"condition":   listing.RecordCondition,
			"prediction":  pred,
			"price":       listing.Price,
			"current_bid": listing.CurrentBid,
			"end_date":    listing.EndDate,
			"url":         fmt.Sprintf("https://www.ebay.com/itm/%s", listing.EbayID),
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i]["prediction"].(float64) > results[j]["prediction"].(float64)
	})

	log.Printf("✅ Returning %d enriched listings for annotation", len(results))

	c.JSON(200, gin.H{
		"listings": results, // All 200
		"stats": gin.H{
			"total_listings":    firstPassResult.TotalProcessed,
			"stage1_candidates": firstPassResult.TopN,
			"stage2_enriched":   enrichedCount,
		},
	})
}

func (h *EbayHandler) GetUnannotatedListings(c *gin.Context) {
	var listings []models.EbayListing

	// Get all listings that haven't been evaluated yet
	result := h.db.Where("evaluated = ?", false).
		Order("end_date ASC").
		Limit(2000). // Start with first 2000
		Find(&listings)

	if result.Error != nil {
		log.Printf("❌ Failed to fetch unannotated listings: %v", result.Error)
		c.JSON(500, gin.H{"error": "Database error"})
		return
	}

	log.Printf("📊 Returning %d unannotated listings", len(listings))

	// Map to response format
	response := []gin.H{}
	for _, listing := range listings {
		response = append(response, gin.H{
			"id":          listing.ID,
			"ebay_id":     listing.EbayID,
			"ebay_title":  listing.EbayTitle,
			"price":       listing.Price,
			"score":       0.0,
			"current_bid": listing.CurrentBid,
			"bid_count":   listing.BidCount,
			"end_date":    listing.EndDate,
		})
	}

	c.JSON(200, gin.H{
		"listings": response,
		"total":    len(response),
	})
}
