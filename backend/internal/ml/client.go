package ml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

type PredictRequest struct {
	Records []MLRecord `json:"records"`
}

type MLRecord struct {
	Artist         string   `json:"artist"`
	Title          string   `json:"title"`
	Label          string   `json:"label"`
	Genres         []string `json:"genres"`
	Styles         []string `json:"styles"`
	Wants          int      `json:"wants"`
	Haves          int      `json:"haves"`
	Year           *int     `json:"year"`
	RecordPrice    string   `json:"record_price"`
	MediaCondition string   `json:"media_condition"`
}

type PredictResponse struct {
	Predictions     []float64 `json:"predictions"`
	MeanPredictions []float64 `json:"mean_predictions"`
	Uncertainties   []float64 `json:"uncertainties"`
	ModelVersion    string    `json:"model_version"`
}

type TrainRequest struct {
	Instances []TrainingInstance `json:"instances"`
}

type TrainingInstance struct {
	ID        uint `json:"id"`
	Predicted bool `json:"predicted"`
	Actual    bool `json:"actual"`
}

type FeedbackRequest struct {
	Records     []MLRecord `json:"records"`
	Labels      []bool     `json:"labels"`
	Predictions []float64  `json:"predictions"`
}

type SelectBatchRequest struct {
	Records         []MLRecord `json:"records"`
	MeanPredictions []float64  `json:"mean_predictions"`
	Uncertainties   []float64  `json:"uncertainties"`
}

type SelectBatchResponse struct {
	SelectedIndices []int `json:"selected_indices"`
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *Client) Predict(records []MLRecord) (*PredictResponse, error) {
	reqBody := PredictRequest{Records: records}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/predict/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var result PredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

func (c *Client) Train(instances []TrainingInstance) error {
	reqBody := TrainRequest{Instances: instances}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/train/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	return nil
}

func (c *Client) SendFeedback(payload map[string]interface{}) error {
	log.Printf("🔵 SendFeedback called with %d records", len(payload["records"].([]map[string]interface{})))

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal feedback: %w", err)
	}

	log.Printf("🔵 Sending to: %s/feedback/", c.baseURL)

	resp, err := c.httpClient.Post(
		c.baseURL+"/feedback/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return fmt.Errorf("failed to send feedback: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	log.Printf("🔵 Response status: %d", resp.StatusCode)
	log.Printf("🔵 Response body: %s", string(body))

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("feedback request failed: %d - %s", resp.StatusCode, string(body))
	}
	return nil
}

func (c *Client) SelectBatch(
	records []MLRecord,
	meanPredictions []float64,
	uncertainties []float64,
) ([]int, error) {
	reqBody := SelectBatchRequest{
		Records:         records,
		MeanPredictions: meanPredictions,
		Uncertainties:   uncertainties,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/select_batch/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	var result SelectBatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response")
	}

	return result.SelectedIndices, nil
}

func (c *Client) RecordPerformance(payload map[string]interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/performance/", // Django endpoint
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

func (c *Client) ScoreListings(records []map[string]interface{}) ([]map[string]interface{}, error) {
	jsonData, _ := json.Marshal(map[string]interface{}{
		"listings": records,
	})

	resp, err := c.httpClient.Post(
		c.baseURL+"/ml/ranking/score/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		ScoredListings []map[string]interface{} `json:"scored_listings"`
	}
	json.Unmarshal(body, &result)

	return result.ScoredListings, nil
}

func (c *Client) TuneWeights(ranking []int64, allListings []int64) error {
	jsonData, _ := json.Marshal(map[string]interface{}{
		"ranking":  ranking,
		"listings": allListings,
	})

	_, err := c.httpClient.Post(
		c.baseURL+"/ml/tune/",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	return err
}
