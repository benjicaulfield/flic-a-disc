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

type BySellerRequest struct {
	Seller string `json:"seller"`
}

type BySellerResponse struct {
	Seller  string         `json:"seller"`
	Total   int            `json:"total"`
	Results []SellerRecord `json:"results"`
}

type SellerRecord struct {
	DiscogsID      string             `json:"discogs_id" gorm:"uniqueIndex:discogs_discogsrecord_discogs_id_key;not null"`
	Artist         string             `json:"artist" gorm:"not null"`
	Title          string             `json:"title" gorm:"not null"`
	Format         models.StringSlice `json:"format" gorm:"type:jsonb;default:'[]'"`
	Label          string             `json:"label" gorm:"type:text"`
	Catno          *string            `json:"catno"`
	Wants          int                `json:"wants" gorm:"default:0"`
	Haves          int                `json:"haves" gorm:"default:0"`
	Genres         models.StringSlice `json:"genres" gorm:"type:jsonb;default:'[]'"`
	Styles         models.StringSlice `json:"styles" gorm:"type:jsonb;default:'[]'"`
	SuggestedPrice float64            `json:"suggested_price"`
	MediaCondition string             `json:"media_condition"`
	Year           *int               `json:"year"`
	Wanted         bool               `json:"wanted" gorm:"default:false"`
	Evaluated      bool               `json:"evaluated" gorm:"default:false"`
	Score          float64            `json:"score"`
	Price          float64            `json:"price"`
	RecordPrice    string             `json:"record_price"`
	Currency       string             `json:"currency"`
}

func (h *Handler) BySellerHandler(c *gin.Context) {
	var req BySellerRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	body, _ := json.Marshal(map[string]interface{}{
		"seller": req.Seller,
	})

	httpReq, err := http.NewRequest("POST", h.GetMLURL()+"/ml/discogs/by-seller/", bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ML service"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 20 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to call ml service"})
		return
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	var mlResponse BySellerResponse
	if err := json.Unmarshal(bodyBytes, &mlResponse); err != nil {
		log.Printf("BySeller decode error: %v — body: %s", err, string(bodyBytes))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse ML response"})
		return
	}

	c.JSON(http.StatusOK, mlResponse)
}
