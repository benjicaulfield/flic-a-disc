package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestMain(m *testing.M) {
	gin.SetMode(gin.TestMode)
	m.Run()
}

func TestKnapsackEndpoint(t *testing.T) {
	reqBody := map[string]interface{}{
		"sellers": []map[string]interface{}{
			{"name": "kim_melody", "shipping_min": 5.0, "currency": "USD"},
		},
		"budget": 250.0,
	}

	body, _ := json.Marshal(reqBody)

	resp, err := http.Post(
		"http://localhost:8001/ml/discogs/knapsack/",
		"application/json",
		bytes.NewBuffer(body),
	)

	assert.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result models.KnapsackResponse
	err = json.NewDecoder(resp.Body).Decode(&result)
	assert.NoError(t, err)

	bodyBytes, _ := io.ReadAll(resp.Body)
	fmt.Println("Response:", string(bodyBytes))

	assert.NotEmpty(t, result.Knapsacks)
	assert.Equal(t, 250.0, result.Budget)
}
