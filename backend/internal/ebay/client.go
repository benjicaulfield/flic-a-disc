package ebay

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type Client struct {
	AppID       string
	CertID      string
	AccessToken string
	BaseURL     string
}

type SearchResponse struct {
	Href          string        `json:"href"`
	Total         int           `json:"total"`
	Limit         int           `json:"limit"`
	Offset        int           `json:"offset"`
	ItemSummaries []ItemSummary `json:"itemSummaries"`
}

type ItemSummary struct {
	ItemID           string            `json:"itemId"`
	Title            string            `json:"title"`
	Price            *Price            `json:"price,omitempty"`
	CurrentBidPrice  *Price            `json:"currentBidPrice,omitempty"`
	LocalizedAspects []LocalizedAspect `json:"localizedAspects,omitempty"`
}

type LocalizedAspect struct {
	Type  string `json:"type"`
	Name  string `json:"name"`
	Value string `json:"value"`
}

type Price struct {
	Value string `json:"value"`
}

func NewClient(appID, certID string) *Client {
	return &Client{
		AppID:   appID,
		CertID:  certID,
		BaseURL: "https://api.ebay.com",
	}
}

func (c *Client) GetAccessToken() error {
	log.Printf("DEBUG AppID length: %d, first 4 chars: %s", len(c.AppID), c.AppID[:min(4, len(c.AppID))])
	log.Printf("DEBUG CertID length: %d", len(c.CertID))
	authURL := "https://api.ebay.com/identity/v1/oauth2/token"
	creds := fmt.Sprintf("%s:%s", c.AppID, c.CertID)
	encodedCreds := base64.StdEncoding.EncodeToString([]byte(creds))

	data := url.Values{}
	data.Set("grant_type", "client_credentials")
	data.Set("scope", "https://api.ebay.com/oauth/api_scope")

	req, err := http.NewRequest("POST", authURL, strings.NewReader(data.Encode()))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Authorization", fmt.Sprintf("Basic %s", encodedCreds))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to get token: %s", string(body))
	}

	var tokenData struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tokenData); err != nil {
		return err
	}

	c.AccessToken = tokenData.AccessToken
	return nil
}

func (c *Client) SearchListings(listingType string, hours int) ([]ItemSummary, error) {
	if err := c.GetAccessToken(); err != nil {
		return nil, err
	}

	const (
		limit     = 200
		maxItems  = 10000
		sleepTime = 200 * time.Millisecond
	)

	sort := "newlyListed"
	filter := fmt.Sprintf("conditionIds:{3000},itemLocationCountry:US,buyingOptions:{%s}", listingType)
	if listingType == "AUCTION" {
		sort = "itemEndDate"
		endTime := time.Now().UTC().Add(24 * time.Hour).Format("2006-01-02T15:04:05.000Z")
		filter += fmt.Sprintf(",itemEndDate:[..%s]", endTime)
	}

	allItems := make([]ItemSummary, 0, maxItems)
	offset := 0
	httpClient := &http.Client{Timeout: 30 * time.Second}

	for offset < maxItems {
		searchURL := fmt.Sprintf("%s/buy/browse/v1/item_summary/search", c.BaseURL)
		req, err := http.NewRequest("GET", searchURL, nil)
		if err != nil {
			return nil, err
		}

		q := req.URL.Query()
		q.Add("q", "lp")
		q.Add("category_ids", "176985")
		q.Add("filter", filter)
		q.Add("limit", strconv.Itoa(limit))
		q.Add("offset", strconv.Itoa(offset))
		q.Add("sort", sort)
		req.URL.RawQuery = q.Encode()

		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.AccessToken))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-EBAY-C-MARKETPLACE-ID", "EBAY_US")
		req.Header.Set("Cache-Control", "no-cache")
		req.Header.Set("Pragma", "no-cache")

		resp, err := httpClient.Do(req)
		if err != nil {
			return nil, err
		}
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, readErr
		}

		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("search failed (offset %d): %s", offset, string(body))
		}

		var sr SearchResponse
		if err := json.Unmarshal(body, &sr); err != nil {
			return nil, fmt.Errorf("failed to decode: %w", err)
		}

		if len(sr.ItemSummaries) == 0 {
			break
		}

		remaining := maxItems - len(allItems)
		if len(sr.ItemSummaries) > remaining {
			sr.ItemSummaries = sr.ItemSummaries[:remaining]
		}

		allItems = append(allItems, sr.ItemSummaries...)
		log.Printf("Fetched %d items (offset %d, total collected %d, total available %d)",
			len(sr.ItemSummaries), offset, len(allItems), sr.Total)

		offset += limit
		time.Sleep(sleepTime)
	}

	return allItems, nil
}

func (c *Client) LookupByItemID(itemID string) (*ItemSummary, error) {
	if c.AccessToken == "" {
		if err := c.GetAccessToken(); err != nil {
			return nil, err
		}
	}

	encodedID := url.QueryEscape(itemID)
	lookupURL := fmt.Sprintf("%s/buy/browse/v1/item/%s", c.BaseURL, encodedID)

	req, err := http.NewRequest("GET", lookupURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.AccessToken))
	req.Header.Set("X-EBAY-C-MARKETPLACE-ID", "EBAY_US")
	req.Header.Set("Accept-Language", "en-US")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("lookup failed: %s", string(body))
	}

	var summary ItemSummary
	if err := json.Unmarshal(body, &summary); err != nil {
		return nil, err
	}

	return &summary, nil
}
