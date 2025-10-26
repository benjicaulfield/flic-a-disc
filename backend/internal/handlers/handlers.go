package handlers

import (
	"log"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"flic-a-disc/internal/config"
	"flic-a-disc/internal/ml"
	"flic-a-disc/internal/models"
	"flic-a-disc/internal/services"
)

type Handler struct {
	db              *gorm.DB
	config          *config.Config
	externalService *services.ExternalService
	mlClient        *ml.Client
}

type ListingResponse struct {
	ID             uint           `json:"id"`
	RecordPrice    string         `json:"record_price"`
	MediaCondition string         `json:"media_condition"`
	Record         RecordResponse `json:"record"`
}

type RecordResponse struct {
	ID             uint               `json:"id"`
	DiscogsID      string             `json:"discogs_id"`
	Artist         string             `json:"artist"`
	Title          string             `json:"title"`
	Label          string             `json:"label"`
	Wants          int                `json:"wants"`
	Haves          int                `json:"haves"`
	Genres         models.StringSlice `json:"genres"`
	Styles         models.StringSlice `json:"styles"`
	SuggestedPrice string             `json:"suggested_price"`
	Year           *int               `json:"year"`
}

func New(db *gorm.DB, cfg *config.Config) *Handler {
	return &Handler{
		db:              db,
		config:          cfg,
		externalService: services.NewExternalService(cfg),
		mlClient:        ml.NewClient("http://localhost:8001/ml"),
	}
}

// GET /discogs-keepers/
func (h *Handler) GetDiscogsKeepersPage(c *gin.Context) {
	log.Println("GetDiscogsKeepersPage called")

	// Step 1: Fetch 1000 candidate records
	var candidates []models.DiscogsListing

	result := h.db.
		Preload("Record").
		Joins("JOIN discogs_discogsrecord ON discogs_discogslisting.record_id = discogs_discogsrecord.id").
		Where("discogs_discogsrecord.evaluated = ?", false).
		Order("RANDOM()").
		Limit(1000).
		Find(&candidates)

	if result.Error != nil {
		log.Printf("Database error: %v", result.Error)
		c.JSON(500, gin.H{"error": "Failed to fetch listings"})
		return
	}

	if len(candidates) == 0 {
		c.JSON(200, gin.H{"listings": []any{}, "count": 0})
		return
	}

	log.Printf("Fetched %d candidate records", len(candidates))

	// Step 2: Prepare ML records for ALL candidates
	var mlRecords []ml.MLRecord
	candidateMap := make(map[int]models.DiscogsListing)

	for i, listing := range candidates {
		candidateMap[i] = listing

		mlRecords = append(mlRecords, ml.MLRecord{
			Artist:         listing.Record.Artist,
			Title:          listing.Record.Title,
			Label:          listing.Record.Label,
			Genres:         []string(listing.Record.Genres),
			Styles:         []string(listing.Record.Styles),
			Wants:          listing.Record.Wants,
			Haves:          listing.Record.Haves,
			Year:           listing.Record.Year,
			RecordPrice:    listing.RecordPrice,
			MediaCondition: listing.MediaCondition,
		})
	}

	log.Printf("Prepared %d ML records, getting predictions", len(mlRecords))

	// Step 3: Get predictions + uncertainties for all 1000
	predictions, err := h.mlClient.Predict(mlRecords)
	if err != nil {
		log.Printf("ML prediction failed: %v", err)
		c.JSON(500, gin.H{"error": "ML prediction failed"})
		return
	}

	// Step 4: Call bandit selection to pick 20
	selected, err := h.mlClient.SelectBatch(
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

	// Step 5: Build response with selected records
	var response []ListingResponse
	var selectedPredictions []float64
	var selectedMeanPredictions []float64
	var selectedUncertainties []float64

	for _, idx := range selected {
		listing := candidateMap[idx]

		response = append(response, ListingResponse{
			ID:             listing.ID,
			RecordPrice:    listing.RecordPrice,
			MediaCondition: listing.MediaCondition,
			Record: RecordResponse{
				ID:             listing.Record.ID,
				DiscogsID:      listing.Record.DiscogsID,
				Artist:         listing.Record.Artist,
				Title:          listing.Record.Title,
				Label:          listing.Record.Label,
				Wants:          listing.Record.Wants,
				Haves:          listing.Record.Haves,
				Genres:         []string(listing.Record.Genres),
				Styles:         []string(listing.Record.Styles),
				SuggestedPrice: listing.Record.SuggestedPrice,
				Year:           listing.Record.Year,
			},
		})

		selectedPredictions = append(selectedPredictions, predictions.Predictions[idx])
		selectedMeanPredictions = append(selectedMeanPredictions, predictions.MeanPredictions[idx])
		selectedUncertainties = append(selectedUncertainties, predictions.Uncertainties[idx])
	}

	c.JSON(200, gin.H{
		"listings":         response,
		"count":            len(response),
		"predictions":      selectedPredictions,
		"mean_predictions": selectedMeanPredictions,
		"uncertainties":    selectedUncertainties,
		"model_version":    predictions.ModelVersion,
	})
}

func (h *Handler) TestDB(c *gin.Context) {
	// Try to ping the database
	sqlDB, err := h.db.DB() // Get underlying *sql.DB
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get DB connection", "details": err.Error()})
		return
	}

	if err := sqlDB.Ping(); err != nil {
		c.JSON(500, gin.H{"error": "Database ping failed", "details": err.Error()})
		return
	}

	c.JSON(200, gin.H{"message": "Database connection successful!"})
}

func (h *Handler) GetStats(c *gin.Context) {
	var totalCount int64
	var labeledCount int64

	if err := h.db.Model(&models.DiscogsListing{}).Count(&totalCount).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to get total count"})
		return
	}

	labeledCount = 0

	c.JSON(200, gin.H{
		"total":   totalCount,
		"labeled": labeledCount,
	})
}

func (h *Handler) LabelRecords(c *gin.Context) {
	var req LabelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}

	// Process each record decision (not listing)
	for _, label := range req.Labels {
		// Get the record_id from the listing
		var listing models.DiscogsListing
		if err := h.db.First(&listing, label.ID).Error; err != nil {
			continue
		}

		// Update the record, not the listing
		if err := h.db.Model(&models.Record{}).
			Where("id = ?", listing.RecordID).
			Updates(map[string]interface{}{
				"wanted":    label.Label,
				"evaluated": true,
			}).Error; err != nil {
			log.Printf("Failed to update record %d: %v", listing.RecordID, err)
		}
	}

	feedbackPayload := map[string]interface{}{
		"records":     req.Records,
		"labels":      extractLabels(req.Labels),
		"predictions": req.Predictions,
	}

	if err := h.mlClient.SendFeedback(feedbackPayload); err != nil {
		log.Printf("Failed to send ML feedback: %v", err)
		// Don't fail the request - labels are still saved
	} else {
		log.Printf("Successfully sent feedback for %d records", len(req.Records))
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

type LabelRequest struct {
	Labels []struct {
		ID    uint `json:"id"`
		Label bool `json:"label"`
	} `json:"labels"`
	Records     []map[string]interface{} `json:"records"`
	Predictions []float64                `json:"predictions"`
}

func (h *Handler) GetWantedRecords(c *gin.Context) {
	var records []models.Record

	result := h.db.Where("evaluated = ?", true).Find(&records)
	if result.Error != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch evaluated records"})
		return
	}

	c.JSON(200, gin.H{
		"evaluated": records,
		"count":     len(records),
	})
}

func (h *Handler) RecordBatchPerformance(c *gin.Context) {
	var req struct {
		Correct int `json:"correct"`
		Total   int `json:"total"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "INVALID"})
		return
	}

	performancePayload := map[string]interface{}{
		"correct": req.Correct,
		"total":   req.Total,
	}

	result, err := h.mlClient.RecordPerformance(performancePayload)
	if err != nil {
		log.Printf("Failed to record performance in ML service: %v", err)
		c.JSON(500, gin.H{"error": "Failed to record performance"})
		return
	}

	c.JSON(200, result)
}
