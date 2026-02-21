package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"github.com/benjicaulfield/flic-a-disc/internal/config"
	"github.com/benjicaulfield/flic-a-disc/internal/ml"
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/benjicaulfield/flic-a-disc/internal/services"
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
	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		mlURL = "http://localhost:8001" // default for local
	}

	return &Handler{
		db:              db,
		config:          cfg,
		externalService: services.NewExternalService(cfg),
		mlClient:        ml.NewClient(mlURL + "/ml"),
	}
}

func (h *Handler) getMLURL() string {
	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		return "http://localhost:8001"
	}
	return mlURL
}

// GET /discogs-keepers/
func (h *Handler) GetDiscogsKeepersPage(c *gin.Context) {
	log.Println("GetDiscogsKeepersPage called")

	var totalLabelled int64
	h.db.Model(&models.Record{}).Where("evaluated = ?", true).Count(&totalLabelled)

	// Step 1: Fetch 1000 candidate records
	var candidates []models.Record

	result := h.db.
		Where("evaluated = ?", false).
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
	candidateMap := make(map[int]models.Record)

	for i, record := range candidates {
		candidateMap[i] = record

		mlRecords = append(mlRecords, ml.MLRecord{
			Artist: record.Artist,
			Title:  record.Title,
			Label:  record.Label,
			Genres: []string(record.Genres),
			Styles: []string(record.Styles),
			Wants:  record.Wants,
			Haves:  record.Haves,
			Year:   record.Year,
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
		h.db.Model(&models.Record{}).
			Where("id IN ?", selectedIDs).
			Update("evaluated", true)
		log.Printf("Marked %d records as evaluated", len(selectedIDs))
	}

	c.JSON(200, gin.H{
		"records":          response,
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

	if err := h.db.Model(&models.Record{}).Count(&totalCount).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to get total count"})
		return
	}

	if err := h.db.Model(&models.Record{}).Where("evaluated = ?", true).Count(&labeledCount).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to get labeled count"})
		return
	}

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

	log.Printf("📥 Received %d labels", len(req.Labels))

	// Process each record decision
	for _, label := range req.Labels {
		log.Printf("Processing label: ID=%d, Label=%v", label.ID, label.Label)

		var listing models.DiscogsListing
		if err := h.db.First(&listing, label.ID).Error; err != nil {
			log.Printf("❌ Listing not found for ID %d: %v", label.ID, err)
			continue
		}
		log.Printf("Found listing %d -> record %d", listing.ID, listing.RecordID)

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
		"records":          req.Records,
		"labels":           extractLabels(req.Labels),
		"predictions":      req.Predictions,
		"mean_predictions": req.MeanPredictions,
		"uncertainties":    req.Uncertainties,
	}

	log.Printf("🔄 Sending feedback to ML service...") // ✅ ADD
	if err := h.mlClient.SendFeedback(feedbackPayload); err != nil {
		log.Printf("❌ Failed to send ML feedback: %v", err) // ✅ ADD
	} else {
		log.Printf("✅ Successfully sent feedback for %d records", len(req.Records)) // ✅ ADD
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
	Records         []map[string]interface{} `json:"records"`
	Predictions     []float64                `json:"predictions"`
	MeanPredictions []float64                `json:"mean_predictions"` // ✅ NEW
	Uncertainties   []float64                `json:"uncertainties"`    // ✅ NEW
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

func (h *Handler) GetTodos(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	// Forward to Django with user_id header
	req, _ := http.NewRequest("GET", h.getMLURL()+"/ml/todos/", nil)
	req.Header.Set("X-User-ID", fmt.Sprint(userID))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch todos"})
		return
	}
	defer resp.Body.Close()

	var todos []map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&todos)
	c.JSON(200, todos)
}

func (h *Handler) CreateTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	var body map[string]interface{}
	c.BindJSON(&body)

	jsonBody, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", "http://localhost:8001/ml/todos/", bytes.NewBuffer(jsonBody))
	req.Header.Set("X-User-ID", fmt.Sprint(userID))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to create todo"})
		return
	}
	defer resp.Body.Close()

	var todo map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&todo)
	c.JSON(resp.StatusCode, todo)
}

func (h *Handler) UpdateTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	todoID := c.Param("id")
	var body map[string]interface{}
	c.BindJSON(&body)

	jsonBody, _ := json.Marshal(body)
	req, _ := http.NewRequest("PATCH", fmt.Sprintf("%s/ml/todos/%s/", h.getMLURL(), todoID), bytes.NewBuffer(jsonBody))
	req.Header.Set("X-User-ID", fmt.Sprint(userID))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to update todo"})
		return
	}
	defer resp.Body.Close()

	var todo map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&todo)
	c.JSON(resp.StatusCode, todo)
}

func (h *Handler) DeleteTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	todoID := c.Param("id")
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/ml/todos/%s/", h.getMLURL(), todoID), nil)
	req.Header.Set("X-User-ID", fmt.Sprint(userID))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to delete todo"})
		return
	}
	defer resp.Body.Close()

	c.Status(resp.StatusCode)
}

func getExchangeRates() (map[string]float64, error) {
	resp, err := http.Get("https://open.er-api.com/v6/latest/USD")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var rateData models.ExchangeRateResponse
	if err := json.NewDecoder(resp.Body).Decode(&rateData); err != nil {
		return nil, err
	}

	return rateData.Rates, nil
}

func (h *Handler) RankingTrainer(c *gin.Context) {
	var session models.RankingSession

	result := h.db.Where("completed = ?", false).First(&session)
	if result.Error != nil {
		var listings []models.DiscogsListing
		listingResult := h.db.
			Joins("Record").
			Where("discogs_discogslisting.evaluated = ? AND discogs_discogsrecord.wanted = ?", true, true).
			Order("RANDOM()").
			Limit(400).
			Find(&listings)

		if listingResult.Error != nil {
			log.Printf("Database error: %v", listingResult.Error)
			c.JSON(500, gin.H{"error": "Failed to fetch listings"})
			return
		}

		mlRecords := make([]map[string]interface{}, len(listings))
		listingIDs := make([]int64, len(listings))
		for i, listing := range listings {
			mlRecords[i] = map[string]interface{}{
				"id":     listing.ID,
				"artist": listing.Record.Artist,
				"title":  listing.Record.Title,
				"label":  listing.Record.Label,
				"genres": listing.Record.Genres,
				"styles": listing.Record.Styles,
				"wants":  listing.Record.Wants,
				"haves":  listing.Record.Haves,
				"year":   listing.Record.Year,
			}
			listingIDs[i] = int64(listing.ID)
		}

		scoredRecords, err := h.mlClient.ScoreListings(mlRecords)
		if err != nil {
			c.JSON(500, gin.H{"error": "ml scoring failed"})
			return
		}

		sort.Slice(scoredRecords, func(i, j int) bool {
			return scoredRecords[i]["score"].(float64) > scoredRecords[j]["score"].(float64)
		})

		// Create session
		session = models.RankingSession{
			ListingIDs: listingIDs,
			Completed:  false,
		}
		h.db.Create(&session)
	}

	var completedBatches int64
	h.db.Model(&models.RankingBatch{}).Where("session_id = ?", session.ID).Count(&completedBatches)

	batchIndex := int(completedBatches)
	totalBatches := len(session.ListingIDs) / 10

	if batchIndex >= totalBatches {
		// Session complete
		session.Completed = true
		h.db.Save(&session)
		c.JSON(200, gin.H{"message": "All batches complete"})
		return
	}

	// Get current batch of 10 listings
	start := batchIndex * 10
	end := start + 10
	batchIDs := session.ListingIDs[start:end]

	var listings []models.DiscogsListing
	h.db.Joins("Record").Where("discogs_discogslisting.id IN ?", batchIDs).Find(&listings)

	// Format for frontend
	records := make([]gin.H, len(listings))
	for i, listing := range listings {
		records[i] = gin.H{
			"id":     listing.ID,
			"artist": listing.Record.Artist,
			"title":  listing.Record.Title,
			"label":  listing.Record.Label,
			"genres": listing.Record.Genres,
			"styles": listing.Record.Styles,
			"wants":  listing.Record.Wants,
			"haves":  listing.Record.Haves,
			"year":   listing.Record.Year,
		}
	}

	c.JSON(200, gin.H{
		"records":       records,
		"batch_index":   batchIndex,
		"total_batches": totalBatches,
	})
}

func (h *Handler) SubmitRanking(c *gin.Context) {
	var req struct {
		Ranking []int64 `json:"ranking"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}

	var session models.RankingSession
	if err := h.db.Where("completed = ?", false).First(&session).Error; err != nil {
		c.JSON(404, gin.H{"error": "No active session"})
		return
	}

	var batchIndex int64
	h.db.Model(&models.RankingBatch{}).Where("session_id = ?", session.ID).Count(&batchIndex)

	batch := models.RankingBatch{
		SessionID:  session.ID,
		BatchIndex: int(batchIndex),
		Ranking:    req.Ranking,
	}
	h.db.Create(&batch)

	// Call ML tune endpoint
	go h.mlClient.TuneWeights(req.Ranking, session.ListingIDs)

	c.JSON(200, gin.H{"success": true})
}

func (h *Handler) KnapsackHandler(c *gin.Context) {
	fmt.Println("=== Knapsack handler called ===")

	var req models.KnapsackRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		fmt.Println("Error binding JSON:", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	fmt.Println("Request received:", req)
	fmt.Println("Seller:", req.Seller) // ADD THIS
	fmt.Println("Budget:", req.Budget)
	fmt.Println("Request received:", req)

	mlReq := map[string]interface{}{
		"seller": req.Seller,
		"budget": req.Budget,
	}
	fmt.Println("building ml request")

	body, _ := json.Marshal(mlReq)

	httpReq, err := http.NewRequest("POST", h.getMLURL()+"/ml/discogs/knapsack/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ML service"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	fmt.Println("calling ml service")

	client := &http.Client{
		Timeout: 20 * time.Minute,
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		fmt.Println("HTTP request error:", err)

		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to call ML service"})
		return
	}
	defer resp.Body.Close()
	fmt.Println("ML service responded with status:", resp.StatusCode)

	var mlResponse models.KnapsackResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResponse); err != nil {
		fmt.Println("Decode error:", err)
		bodyBytes, _ := io.ReadAll(resp.Body)
		fmt.Println("Response body:", string(bodyBytes))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse ML response"})
		return
	}

	c.JSON(http.StatusOK, mlResponse)
}
