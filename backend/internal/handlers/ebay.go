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
	ebayClient      *ebay.Client
	db              *gorm.DB
	mu              sync.RWMutex
	cachedAuctions  []FilteredListing
	cachedBIN       []FilteredListing
	fetchInProgress bool
}

func NewEbayHandler(appID, certID string, db *gorm.DB) *EbayHandler {
	h := &EbayHandler{
		ebayClient: ebay.NewClient(appID, certID),
		db:         db,
	}
	return h
}

type Price struct {
	Value string `json:"value"`
}

type RawListing struct {
	EbayID    string `json:"ebay_id"`
	EbayTitle string `json:"ebay_title"`
	Price     string `json:"price"`
}

type FilteredListing struct {
	EbayID    string  `json:"ebay_id"`
	EbayTitle string  `json:"ebay_title"`
	Price     string  `json:"price"`
	Score     float64 `json:"tfidf_score"` // or whatever the ML service calls it
}

type ScoredListing struct {
	EbayID     string  `json:"ebay_id"`
	EbayTitle  string  `json:"ebay_title"`
	Price      string  `json:"price"`
	TfidfScore float64 `json:"tfidf_score"`
}

type filterRequest struct {
	Listings []RawListing `json:"listings"`
	TopN     int          `json:"top_n"`
}

var result struct {
	TopListings []FilteredListing `json:"top_listings"`
}

func (h *EbayHandler) fetchAndCacheAuctions() error {
	allResults, err := h.ebayClient.SearchListings("AUCTION", 48)
	if err != nil {
		log.Printf("Failed to fetch listings: %v", err)
		return err
	}
	var raw []RawListing
	for _, item := range allResults {
		var price string
		if item.CurrentBidPrice != nil {
			price = item.CurrentBidPrice.Value
		} else if item.Price != nil {
			price = item.Price.Value
		}
		raw = append(raw, RawListing{
			EbayID:    item.ItemID,
			EbayTitle: item.Title,
			Price:     price,
		})
	}

	log.Printf("Found %d raw listings from ebay", len(raw))

	filteredListings, allScored, err := h.filterListingsByTFIDF(raw)
	if err != nil {
		log.Printf("Failed to filter listings: %v", err)
		return err
	}

	if err := h.saveToCSV(allScored); err != nil {
		log.Printf("failed to write auction CSV")
	}

	h.mu.Lock()
	h.cachedAuctions = filteredListings
	h.mu.Unlock()

	log.Printf("✅ Cached %d filtered listings in memory", len(filteredListings))
	return nil
}

func (h *EbayHandler) fetchAndCacheBuyItNows() error {
	allResults, err := h.ebayClient.SearchListings("FIXED_PRICE", 0)
	if err != nil {
		log.Printf("Failed to fetch listings: %v", err)
		return err
	}
	var raw []RawListing
	for _, item := range allResults {
		var price string
		if item.CurrentBidPrice != nil {
			price = item.CurrentBidPrice.Value
		} else if item.Price != nil {
			price = item.Price.Value
		}
		raw = append(raw, RawListing{
			EbayID:    item.ItemID,
			EbayTitle: item.Title,
			Price:     price,
		})
	}

	log.Printf("Found %d raw listings from ebay", len(raw))

	filteredListings, allScored, err := h.filterListingsByTFIDF(raw)
	if err != nil {
		log.Printf("Failed to filter listings: %v", err)
		return err
	}

	if err := h.saveToCSV(allScored); err != nil {
		log.Printf("failed to write auction CSV")
	}

	h.mu.Lock()
	h.cachedBIN = filteredListings
	h.mu.Unlock()

	log.Printf("✅ Cached %d filtered listings in memory", len(filteredListings))
	return nil
}

func (h *EbayHandler) filterListingsByTFIDF(listings []RawListing) ([]FilteredListing, []ScoredListing, error) {
	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		mlURL = "http://localhost:8001"
	}
	filterURL := mlURL + "/ml/ebay_title_similarity_filter/"

	if len(listings) > 0 {
		b, _ := json.Marshal(listings[0])
		log.Printf("sample listing JSON: %s", string(b))
	}

	jsonBody, err := json.Marshal(filterRequest{
		Listings: listings,
		TopN:     1000,
	})

	if err != nil {
		return nil, nil, err
	}

	resp, err := http.Post(filterURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	var result struct {
		TopListings []FilteredListing `json:"top_listings"`
		AllScored   []ScoredListing   `json:"all_scored"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, nil, err
	}
	log.Printf("decoded %d top listings, %d all scored", len(result.TopListings), len(result.AllScored))

	return result.TopListings, result.AllScored, nil
}

func (h *EbayHandler) TriggerFetchAuctions(c *gin.Context) {
	h.mu.Lock()

	if h.fetchInProgress {
		h.mu.Unlock()
		c.JSON(409, gin.H{"message": "Fetch already in progress"})
		return
	}

	h.fetchInProgress = true
	h.mu.Unlock()
	h.fetchAndCacheAuctions()
	h.mu.Lock()
	h.fetchInProgress = false
	h.mu.Unlock()

	c.JSON(200, gin.H{"message": "Fetch complete"})
}

func (h *EbayHandler) TriggerFetchBuyItNows(c *gin.Context) {
	h.mu.Lock()

	if h.fetchInProgress {
		h.mu.Unlock()
		c.JSON(409, gin.H{"message": "Fetch already in progress"})
		return
	}

	h.fetchInProgress = true
	h.mu.Unlock()
	h.fetchAndCacheBuyItNows()
	h.mu.Lock()
	h.fetchInProgress = false
	h.mu.Unlock()

	c.JSON(200, gin.H{"message": "Fetch complete"})
}

func (h *EbayHandler) GetCachedBIN(c *gin.Context) {
	h.mu.Lock()
	listings := h.cachedBIN
	h.mu.Unlock()
	c.JSON(200, gin.H{"listings": listings})
}

func (h *EbayHandler) GetCachedAuctions(c *gin.Context) {
	h.mu.Lock()
	listings := h.cachedAuctions
	h.mu.Unlock()
	c.JSON(200, gin.H{"listings": listings})
}

func (h *EbayHandler) saveToCSV(listings []ScoredListing) error {
	filename := time.Now().Format("ebay_buyitnow_2006-01-02.csv")
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	writer.Write([]string{"ebay_id", "ebay_title", "price", "tfidf_score"})

	// Data
	for _, l := range listings {
		writer.Write([]string{
			l.EbayID,
			l.EbayTitle,
			l.Price,
			fmt.Sprintf("%.4f", l.TfidfScore),
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

	saved := 0
	for _, ebayID := range req.EbayIDs {
		item, err := h.ebayClient.LookupByItemID(ebayID)
		if err != nil {
			log.Printf("Failed to lookup %s: %v", ebayID, err)
			continue
		}

		artist, album, label, format, year, recordCondition, genre, style := parseRecordMetadata(item)

		price := ""
		if item.Price != nil {
			price = item.Price.Value
		}

		listing := models.EbayListing{
			EbayID:         ebayID,
			EbayTitle:      item.Title,
			Price:          price,
			Artist:         artist,
			Title:          album,
			Label:          label,
			Format:         models.StringSlice{format},
			Year:           year,
			MediaCondition: recordCondition,
			Genre:          genre,
			Style:          style,
		}

		if err := h.db.Where(models.EbayListing{EbayID: ebayID}).
			FirstOrCreate(&listing).Error; err != nil {
			log.Printf("Failed to save listing %s: %v", ebayID, err)
			continue
		}
		saved++
	}

	c.JSON(200, gin.H{"message": "Saved", "count": saved})
}

func parseRecordMetadata(item *ebay.ItemSummary) (artist, album, label, format, year, recordCondition, genre, style string) {
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
		artist, album, label, format, year, recordCondition, genre, style :=
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
			"id":         listing.ID,
			"ebay_id":    listing.EbayID,
			"title":      listing.EbayTitle,
			"artist":     listing.Artist,
			"album":      listing.Title,
			"label":      listing.Label,
			"genre":      listing.Genre,
			"style":      listing.Style,
			"year":       listing.Year,
			"format":     listing.Format,
			"condition":  listing.MediaCondition,
			"prediction": pred,
			"price":      listing.Price,
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
