package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
)

func (h *Handler) KnapsackHandler(c *gin.Context) {
	var req models.KnapsackRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	body, _ := json.Marshal(map[string]interface{}{
		"seller": req.Seller,
		"budget": req.Budget,
	})

	httpReq, err := http.NewRequest("POST", h.GetMLURL()+"/ml/discogs/knapsack/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ML service"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 20 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Knapsack ML service error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to call ML service"})
		return
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	var mlResponse models.KnapsackResponse
	if err := json.Unmarshal(bodyBytes, &mlResponse); err != nil {
		log.Printf("Knapsack decode error: %v — body: %s", err, string(bodyBytes))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse ML response"})
		return
	}

	c.JSON(http.StatusOK, mlResponse)
}

func (h *Handler) SellerBrowse(c *gin.Context) {
	mlURL := h.GetMLURL() + "/ml/discogs/seller/browse/"
	if q := c.Request.URL.RawQuery; q != "" {
		mlURL += "?" + q
	}
	resp, err := http.Get(mlURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

func (h *Handler) KnapsackSessionsList(c *gin.Context) {
	mlURL := h.GetMLURL() + "/ml/discogs/knapsack/sessions/"
	if q := c.Request.URL.RawQuery; q != "" {
		mlURL += "?" + q
	}
	resp, err := http.Get(mlURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

func (h *Handler) KnapsackSessionUpdate(c *gin.Context) {
	id := c.Param("id")
	mlURL := h.GetMLURL() + "/ml/discogs/knapsack/sessions/" + id + "/"
	bodyBytes, _ := io.ReadAll(c.Request.Body)
	req, err := http.NewRequest("PATCH", mlURL, bytes.NewBuffer(bodyBytes))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to build request"})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

func (h *Handler) KnapsackSessionsCompare(c *gin.Context) {
	mlURL := h.GetMLURL() + "/ml/discogs/knapsack/sessions/compare/"
	if q := c.Request.URL.RawQuery; q != "" {
		mlURL += "?" + q
	}
	resp, err := http.Get(mlURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "ML service unavailable"})
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, "application/json", body)
}

func GetExchangeRates() (map[string]float64, error) {
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
