package handlers

import (
	"bytes"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

func (h *Handler) TestDB(c *gin.Context) {
	sqlDB, err := h.DB.DB()
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

func (h *Handler) proxyPost(c *gin.Context, path string) {
	body, _ := io.ReadAll(c.Request.Body)
	req, _ := http.NewRequest("POST", h.GetMLURL()+path, bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 20 * time.Minute}).Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", respBody)
}
