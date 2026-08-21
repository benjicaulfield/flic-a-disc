package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/benjicaulfield/flic-a-disc/internal/ml"
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
)

// GET /discogs-keepers/
func (h *Handler) GetDiscogsKeepersPage(c *gin.Context) {
	log.Println("GetDiscogsKeepersPage called")

	var totalLabelled int64
	h.DB.Model(&models.DiscogsRecord{}).Where("evaluated = ?", true).Count(&totalLabelled)

	// Step 1: Fetch 1000 candidate records
	var candidates []models.DiscogsRecord

	result := h.DB.
		Where("evaluated = ? AND wants > haves", false).
		Order("RANDOM()").
		Limit(1000).
		Find(&candidates)

	if result.Error != nil {
		log.Printf("Database error: %v", result.Error)
		c.JSON(500, gin.H{"error": "Failed to fetch listings"})
		return
	}

	if len(candidates) == 0 {
		c.JSON(200, gin.H{"records": []any{}, "count": 0})
		return
	}

	log.Printf("Fetched %d candidate records", len(candidates))

	// Step 2: Prepare ML records for ALL candidates
	var mlRecords []ml.MLRecord
	candidateMap := make(map[int]models.DiscogsRecord)

	for i, record := range candidates {
		candidateMap[i] = record

		mlRecords = append(mlRecords, ml.MLRecord{
			Artist:         record.Artist,
			Title:          record.Title,
			Label:          record.Label,
			Genres:         []string(record.Genres),
			Styles:         []string(record.Styles),
			Wants:          record.Wants,
			Haves:          record.Haves,
			Year:           record.Year,
			SuggestedPrice: record.SuggestedPrice,
		})
	}

	log.Printf("Prepared %d ML records, getting predictions", len(mlRecords))

	// Step 3: Get predictions + uncertainties for all 1000
	predictions, err := h.MLClient.Predict(mlRecords)
	if err != nil {
		log.Printf("ML prediction failed: %v", err)
		c.JSON(500, gin.H{"error": "ML prediction failed"})
		return
	}

	selected, err := h.MLClient.SelectBatch(
		mlRecords,
		predictions.MeanPredictions,
		predictions.Uncertainties,
	)
	if err != nil {
		log.Printf("Bandit selection failed: %v", err)
		c.JSON(500, gin.H{"error": "Bandit selection failed"})
		return
	}

	log.Printf("Bandit selected %d records", len(selected))

	var selectedIDs []uint
	for _, idx := range selected {
		selectedIDs = append(selectedIDs, candidateMap[idx].ID)
	}

	var response []RecordResponse
	var selectedPredictions []float64
	var selectedMeanPredictions []float64
	var selectedUncertainties []float64

	for _, idx := range selected {
		record := candidateMap[idx]

		response = append(response, RecordResponse{
			ID:             record.ID,
			DiscogsID:      record.DiscogsID,
			Artist:         record.Artist,
			Title:          record.Title,
			Label:          record.Label,
			Wants:          record.Wants,
			Haves:          record.Haves,
			Genres:         []string(record.Genres),
			Styles:         []string(record.Styles),
			SuggestedPrice: record.SuggestedPrice,
			Year:           record.Year,
		})

		selectedPredictions = append(selectedPredictions, predictions.Predictions[idx])
		selectedMeanPredictions = append(selectedMeanPredictions, predictions.MeanPredictions[idx])
		selectedUncertainties = append(selectedUncertainties, predictions.Uncertainties[idx])
	}

	if len(selectedIDs) > 0 {
		h.DB.Model(&models.DiscogsRecord{}).
			Where("id IN ?", selectedIDs).
			Update("evaluated", true)
		log.Printf("Marked %d records as evaluated", len(selectedIDs))
	}

	var missingPriceIDs []string
	for _, r := range response {
		if r.SuggestedPrice == "" {
			missingPriceIDs = append(missingPriceIDs, r.DiscogsID)
		}
	}
	log.Printf("Records missing suggested_price: %d / %d", len(missingPriceIDs), len(response))
	if len(missingPriceIDs) > 0 {
		log.Printf("Fetching suggested price for %d records: %v", len(missingPriceIDs), missingPriceIDs[:min(3, len(missingPriceIDs))])
		enriched, err := h.MLClient.EnrichRecords(missingPriceIDs)
		if err != nil {
			log.Printf("Enrich error: %v", err)
		} else {
			log.Printf("Enrich returned %d results", len(enriched))
			enrichMap := make(map[string]ml.EnrichedRecord, len(enriched))
			for _, e := range enriched {
				enrichMap[e.DiscogsID] = e
			}
			for i, r := range response {
				if e, ok := enrichMap[r.DiscogsID]; ok {
					response[i].SuggestedPrice = e.SuggestedPrice
				}
			}
		}
	}

	c.JSON(200, gin.H{
		"records":          response,
		"count":            len(response),
		"predictions":      selectedPredictions,
		"mean_predictions": selectedMeanPredictions,
		"uncertainties":    selectedUncertainties,
		"model_version":    predictions.ModelVersion,
		"threshold":        predictions.Threshold,
	})
}

type LabelRequest struct {
	Labels []struct {
		ID    uint `json:"id"`
		Label bool `json:"label"`
	} `json:"labels"`
	Records         []map[string]interface{} `json:"records"`
	Predictions     []float64                `json:"predictions"`
	MeanPredictions []float64                `json:"mean_predictions"`
	Uncertainties   []float64                `json:"uncertainties"`
}

func (h *Handler) LabelRecords(c *gin.Context) {
	var req LabelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}

	log.Printf("📥 Received %d labels", len(req.Labels))

	// Process each record decision
	for _, label := range req.Labels {
		log.Printf("Processing label: ID=%d, Label=%v", label.ID, label.Label)

		if err := h.DB.Model(&models.DiscogsRecord{}).
			Where("id = ?", label.ID).
			Updates(map[string]interface{}{
				"wanted":    label.Label,
				"evaluated": true,
			}).Error; err != nil {
			log.Printf("Failed to update record %d: %v", label.ID, err)
		}
	}

	feedbackPayload := map[string]interface{}{
		"records":          req.Records,
		"labels":           extractLabels(req.Labels),
		"predictions":      req.Predictions,
		"mean_predictions": req.MeanPredictions,
		"uncertainties":    req.Uncertainties,
	}

	log.Printf("🔄 Sending feedback to ML service...")
	if err := h.MLClient.SendFeedback(feedbackPayload); err != nil {
		log.Printf("❌ Failed to send ML feedback: %v", err)
	} else {
		log.Printf("✅ Successfully sent feedback for %d records", len(req.Records))
	}

	c.JSON(200, gin.H{"message": "Records labeled successfully"})
}

func extractLabels(labels []struct {
	ID    uint `json:"id"`
	Label bool `json:"label"`
}) []bool {
	result := make([]bool, len(labels))
	for i, label := range labels {
		result[i] = label.Label
	}
	return result
}

// GET /api/discogs/oof?offset=0
func (h *Handler) GetOOFBatch(c *gin.Context) {
	offset := c.DefaultQuery("offset", "0")
	mlURL := h.GetMLURL()

	resp, err := http.Get(mlURL + "/ml/discogs/oof/?offset=" + offset)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch OOF batch"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		c.JSON(500, gin.H{"error": "Failed to parse OOF batch"})
		return
	}
	c.JSON(200, result)
}

type CatalogLabelRequest struct {
	Labels []struct {
		ID    uint `json:"id"`
		Label bool `json:"label"`
	} `json:"labels"`
	Records         []map[string]interface{} `json:"records"`
	Predictions     []float64                `json:"predictions"`
	MeanPredictions []float64                `json:"mean_predictions"`
	Uncertainties   []float64                `json:"uncertainties"`
}

// POST /api/discogs/catalog/labels
func (h *Handler) LabelCatalogRecords(c *gin.Context) {
	var req CatalogLabelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}

	log.Printf("📥 Received %d catalog labels", len(req.Labels))

	recordMap := make(map[int]map[string]interface{})
	for i, record := range req.Records {
		recordMap[i] = record
	}

	for i, label := range req.Labels {
		recordData := recordMap[i]
		if recordData == nil {
			log.Printf("⚠️  No record data for label index %d", i)
			continue
		}

		discogsID, _ := recordData["release_id"].(string)
		if discogsID == "" {
			log.Printf("⚠️  Empty release_id at index %d", i)
			continue
		}

		var existingRecord models.DiscogsRecord
		if err := h.DB.Where("discogs_id = ?", discogsID).First(&existingRecord).Error; err == nil {
			h.DB.Model(&existingRecord).Updates(map[string]interface{}{
				"wanted":       label.Label,
				"evaluated":    true,
				"api_enriched": true,
			})
			log.Printf("✅ Updated existing record: %s (wanted=%v)", discogsID, label.Label)
			continue
		}

		artist, _ := recordData["artist"].(string)
		title, _ := recordData["title"].(string)
		labelName, _ := recordData["label"].(string)
		catno, _ := recordData["catalog_number"].(string)
		suggestedPrice, _ := recordData["suggested_price"].(string)

		wants, haves, year := 0, 0, 0
		if w, ok := recordData["wants"].(float64); ok {
			wants = int(w)
		}
		if hv, ok := recordData["haves"].(float64); ok {
			haves = int(hv)
		}
		if y, ok := recordData["year"].(float64); ok {
			year = int(y)
		} else if yStr, ok := recordData["year"].(string); ok {
			var yInt int
			if _, err := fmt.Sscanf(yStr, "%d", &yInt); err == nil {
				year = yInt
			}
		}

		var genres, styles []string
		if g, ok := recordData["genre"].([]interface{}); ok {
			for _, v := range g {
				if s, ok := v.(string); ok {
					genres = append(genres, s)
				}
			}
		}
		if s, ok := recordData["style"].([]interface{}); ok {
			for _, v := range s {
				if str, ok := v.(string); ok {
					styles = append(styles, str)
				}
			}
		}

		var catnoPtr *string
		if catno != "" {
			catnoPtr = &catno
		}
		var yearPtr *int
		if year != 0 {
			yearPtr = &year
		}

		record := models.DiscogsRecord{
			DiscogsID:      discogsID,
			Artist:         artist,
			Title:          title,
			Format:         models.StringSlice{"Vinyl", "LP"},
			Label:          labelName,
			Catno:          catnoPtr,
			Wants:          wants,
			Haves:          haves,
			Genres:         models.StringSlice(genres),
			Styles:         models.StringSlice(styles),
			SuggestedPrice: suggestedPrice,
			Year:           yearPtr,
			Wanted:         label.Label,
			Evaluated:      true,
		}

		if err := h.DB.Create(&record).Error; err != nil {
			log.Printf("❌ Failed to create record for %s: %v", discogsID, err)
			continue
		}
		log.Printf("✅ Created record: %s - %s (wanted=%v)", artist, title, label.Label)
	}

	keepers := countKeepers(req.Labels)
	log.Printf("✅ Catalog records labeled successfully (%d keepers, %d non-keepers)", keepers, len(req.Labels)-keepers)
	c.JSON(200, gin.H{"message": "Catalog records labeled successfully"})
}

func countKeepers(labels []struct {
	ID    uint `json:"id"`
	Label bool `json:"label"`
}) int {
	count := 0
	for _, label := range labels {
		if label.Label {
			count++
		}
	}
	return count
}

// GET /api/discogs/catalog-candidates
func (h *Handler) GetCatalogCandidates(c *gin.Context) {
	resp, err := http.Get(h.GetMLURL() + "/ml/discogs/catalog-candidates/")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

// POST /api/discogs/catalog-candidates
func (h *Handler) SaveCatalogCandidates(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request"})
		return
	}

	req, err := http.NewRequest("POST", h.GetMLURL()+"/ml/discogs/catalog-candidates/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to build request"})
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("ML service error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", respBody)
}
