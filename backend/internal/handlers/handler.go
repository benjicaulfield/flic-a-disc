package handlers

import (
	"os"

	"gorm.io/gorm"

	"github.com/benjicaulfield/flic-a-disc/internal/config"
	"github.com/benjicaulfield/flic-a-disc/internal/ml"
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/benjicaulfield/flic-a-disc/internal/services"
)

type Handler struct {
	DB              *gorm.DB
	Config          *config.Config
	ExternalService *services.ExternalService
	MLClient        *ml.Client
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
		DB:              db,
		Config:          cfg,
		ExternalService: services.NewExternalService(cfg),
		MLClient:        ml.NewClient(mlURL + "/ml"),
	}
}

func (h *Handler) GetMLURL() string {
	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		return "http://localhost:8001"
	}
	return mlURL
}
