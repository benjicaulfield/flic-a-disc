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
	ebayClient     *ebay.Client
	db             *gorm.DB
	mu             sync.RWMutex
	cachedListings []gin.H
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
	log.Println("🚀 fetchAndCacheListings() CALLED")

	allResults, err := h.ebayClient.SearchAuctionsEndingSoon(48)
	if err != nil {
		log.Printf("Failed to fetch listings: %v", err)
		return
	}

	now := time.Now()
	cutoffStart := now.Add(24 * time.Hour) // 24 hours from now
	cutoffEnd := now.Add(48 * time.Hour)   // 48 hours from now

	var raw []gin.H
	for _, item := range allResults {
		endTime, err := time.Parse(time.RFC3339, item.ItemEndDate)
		if err != nil {
			log.Printf("Failed to parse end time for item %s: %v", item.ItemID, err)
			continue
		}
		if endTime.After(cutoffStart) && endTime.Before(cutoffEnd) {
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

	h.mu.Lock()
	h.cachedListings = filteredListings
	h.mu.Unlock()
	log.Printf("✅ Cached %d filtered listings in memory", len(filteredListings))
}

// ✅ New helper function to call Python filter
func (h *EbayHandler) filterListingsByTFIDF(listings []gin.H) ([]gin.H, error) {
	filterURL := "http://localhost:8001/ml/ebay_title_similarity_filter/"

	requestBody := map[string]interface{}{
		"listings": listings,
		"top_n":    1000,
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
		TopListings []gin.H `json:"top_listings"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.TopListings, nil
}

func (h *EbayHandler) TriggerFetch(c *gin.Context) {
	var fetchInProgress bool
	h.mu.Lock()
	if fetchInProgress {
		h.mu.Unlock()
		c.JSON(409, gin.H{"message": "Fetch already in progress"})
		return
	}
	fetchInProgress = true
	h.mu.Unlock()

	defer func() {
		h.mu.Lock()
		fetchInProgress = false
		h.mu.Unlock()
	}()

	log.Printf("TriggerFetch called from %s", c.Request.RemoteAddr)
	h.fetchAndCacheListings()
	c.JSON(200, gin.H{"message": "Fetch complete"})
}

func (h *EbayHandler) GetListings(c *gin.Context) {
	var listings []models.EbayListing
	if err := h.db.Where("evaluated = ?", false).
		Order("end_date ASC").
		Find(&listings).Error; err != nil {
		c.JSON(500, gin.H{"error": "Database error"})
		return
	}
	c.JSON(200, gin.H{"listings": listings})
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

func (h *EbayHandler) SaveKeepers(c *gin.Context) {
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
			Keeper:          true,
		}

		h.db.Create(&listing)
	}

	c.JSON(200, gin.H{"message": "Saved", "count": len(req.EbayIDs)})
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
	h.mu.RLock()
	cached := h.cachedListings
	h.mu.RUnlock()

	if len(cached) == 0 {
		c.JSON(200, gin.H{
			"listings": []gin.H{},
			"total":    0,
			"message":  "No listings cached. Click Refresh to fetch from eBay.",
		})
		return
	}

	log.Printf("📊 Returning %d cached listings", len(cached))

	c.JSON(200, gin.H{
		"listings": cached,
		"total":    len(cached),
	})
}
