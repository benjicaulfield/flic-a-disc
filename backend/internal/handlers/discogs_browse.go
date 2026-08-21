package handlers

import (
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
)

func (h *Handler) GetWantedRecords(c *gin.Context) {
	var records []models.DiscogsRecord

	result := h.DB.Where("evaluated = ?", true).Find(&records)
	if result.Error != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch evaluated records"})
		return
	}

	c.JSON(200, gin.H{
		"evaluated": records,
		"count":     len(records),
	})
}

func (h *Handler) GetStats(c *gin.Context) {
	var totalCount int64
	var labeledCount int64

	if err := h.DB.Model(&models.DiscogsRecord{}).Count(&totalCount).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to get total count"})
		return
	}

	if err := h.DB.Model(&models.DiscogsRecord{}).Where("evaluated = ?", true).Count(&labeledCount).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to get labeled count"})
		return
	}

	c.JSON(200, gin.H{
		"total":   totalCount,
		"labeled": labeledCount,
	})
}
