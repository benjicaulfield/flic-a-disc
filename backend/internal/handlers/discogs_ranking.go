package handlers

import (
	"log"
	"sort"

	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
)

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

	result, err := h.MLClient.RecordPerformance(performancePayload)
	if err != nil {
		log.Printf("Failed to record performance in ML service: %v", err)
		c.JSON(500, gin.H{"error": "Failed to record performance"})
		return
	}

	c.JSON(200, result)
}

func (h *Handler) RankingTrainer(c *gin.Context) {
	var session models.RankingSession

	result := h.DB.Where("completed = ?", false).First(&session)
	if result.Error != nil {
		var listings []models.DiscogsListing
		listingResult := h.DB.
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

		scoredRecords, err := h.MLClient.ScoreListings(mlRecords)
		if err != nil {
			c.JSON(500, gin.H{"error": "ml scoring failed"})
			return
		}

		sort.Slice(scoredRecords, func(i, j int) bool {
			return scoredRecords[i]["score"].(float64) > scoredRecords[j]["score"].(float64)
		})

		session = models.RankingSession{
			ListingIDs: listingIDs,
			Completed:  false,
		}
		h.DB.Create(&session)
	}

	var completedBatches int64
	h.DB.Model(&models.RankingBatch{}).Where("session_id = ?", session.ID).Count(&completedBatches)

	batchIndex := int(completedBatches)
	totalBatches := len(session.ListingIDs) / 10

	if batchIndex >= totalBatches {
		session.Completed = true
		h.DB.Save(&session)
		c.JSON(200, gin.H{"message": "All batches complete"})
		return
	}

	start := batchIndex * 10
	end := start + 10
	batchIDs := session.ListingIDs[start:end]

	var listings []models.DiscogsListing
	h.DB.Joins("Record").Where("discogs_discogslisting.id IN ?", batchIDs).Find(&listings)

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
	if err := h.DB.Where("completed = ?", false).First(&session).Error; err != nil {
		c.JSON(404, gin.H{"error": "No active session"})
		return
	}

	var batchIndex int64
	h.DB.Model(&models.RankingBatch{}).Where("session_id = ?", session.ID).Count(&batchIndex)

	batch := models.RankingBatch{
		SessionID:  session.ID,
		BatchIndex: int(batchIndex),
		Ranking:    req.Ranking,
	}
	h.DB.Create(&batch)

	go h.MLClient.TuneWeights(req.Ranking, session.ListingIDs)

	c.JSON(200, gin.H{"success": true})
}
