package services

import (
	"net/http"
	"time"

	"flic-a-disc/internal/config"
)

type ExternalService struct {
	config     *config.Config
	httpClient *http.Client
}

func NewExternalService(cfg *config.Config) *ExternalService {
	return &ExternalService{
		config: cfg,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}
