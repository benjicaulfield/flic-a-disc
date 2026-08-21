package handlers

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"net/url"
	"time"

	"github.com/gin-gonic/gin"
)

// POST /discogs/by-seller
func (h *Handler) BySellerHandler(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request"})
		return
	}

	httpReq, err := http.NewRequest("POST", h.GetMLURL()+"/ml/discogs/by-seller/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ML service"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Scrapes can run up to 25 minutes on the ML side; give this a little headroom
	// above that so the frontend's own timeout is what fires first, not this one.
	client := &http.Client{Timeout: 26 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ml service"})
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	// Passed through as raw bytes rather than decoded into a typed struct — a typed
	// intermediary silently drops any field it doesn't know about (this is how
	// listing_id/wantlist previously vanished between the ML service and the frontend).
	c.Data(resp.StatusCode, "application/json", respBody)
}

// GET /discogs/by-seller/saved?seller=
func (h *Handler) BySellerSavedHandler(c *gin.Context) {
	seller := c.Query("seller")

	resp, err := http.Get(h.GetMLURL() + "/ml/discogs/by-seller/saved/?seller=" + url.QueryEscape(seller))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

// POST /discogs/annotate
func (h *Handler) DiscogsAnnotateHandler(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request"})
		return
	}

	httpReq, err := http.NewRequest("POST", h.GetMLURL()+"/ml/discogs/annotate/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to build request"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 20 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("ML service error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", respBody)
}
