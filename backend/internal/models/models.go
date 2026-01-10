package models

import (
	"database/sql/driver"
	"encoding/json"
	"time"
)

// StringSlice is a custom type for handling JSON arrays in PostgreSQL
type StringSlice []string

func (s StringSlice) Value() (driver.Value, error) {
	if len(s) == 0 {
		return "[]", nil
	}
	return json.Marshal(s)
}

func (s *StringSlice) Scan(value interface{}) error {
	if value == nil {
		*s = StringSlice{}
		return nil
	}

	switch v := value.(type) {
	case []byte:
		return json.Unmarshal(v, s)
	case string:
		return json.Unmarshal([]byte(v), s)
	}
	return nil
}

// Record represents a music record
type Record struct {
	ID             uint        `json:"id" gorm:"primaryKey"`
	DiscogsID      string      `json:"discogs_id" gorm:"uniqueIndex:discogs_discogsrecord_discogs_id_key;not null"`
	Artist         string      `json:"artist" gorm:"not null"`
	Title          string      `json:"title" gorm:"not null"`
	Format         string      `json:"format" gorm:"default:''"`
	Label          string      `json:"label" gorm:"type:text"`
	Catno          *string     `json:"catno"`
	Wants          int         `json:"wants" gorm:"default:0"`
	Haves          int         `json:"Shaves" gorm:"default:0"`
	Added          time.Time   `json:"added" gorm:"default:CURRENT_TIMESTAMP"`
	Genres         StringSlice `json:"genres" gorm:"type:jsonb;default:'[]'"`
	Styles         StringSlice `json:"styles" gorm:"type:jsonb;default:'[]'"`
	SuggestedPrice string      `json:"suggested_price" gorm:"default:''"`
	Year           *int        `json:"year"`
	Wanted         bool        `json:"wanted" gorm:"default:false"`
	Evaluated      bool        `json:"evaluated" gorm:"default:false"`
}

// Seller represents a record seller
type DiscogsSeller struct {
	ID          uint    `json:"id" gorm:"primaryKey"`
	Name        string  `json:"name" gorm:"not null"`
	ShippingMin float64 `json:"shipping_min"`
	Currency    string  `json:"currency"`
}

// Listing represents a record listing by a seller
type DiscogsListing struct {
	ID             uint          `json:"id" gorm:"primaryKey"`
	SellerID       uint          `json:"seller_id" gorm:"not null"`
	Seller         DiscogsSeller `json:"seller" gorm:"foreignKey:SellerID"`
	RecordID       uint          `json:"record_id" gorm:"not null"`
	Record         Record        `json:"record" gorm:"foreignKey:RecordID"`
	RecordPrice    string        `json:"record_price" gorm:"not null"`
	MediaCondition string        `json:"media_condition" gorm:"not null"`
}

type KnapsackItem struct {
	DiscogsID      string   `json:"discogs_id"`
	Artist         string   `json:"artist"`
	Title          string   `json:"title"`
	Price          float64  `json:"price"`
	Currency       string   `json:"currency"`
	MediaCondition string   `json:"media_condition"`
	Seller         string   `json:"seller"`
	Label          string   `json:"label"`
	Catno          string   `json:"catno"`
	Wants          int      `json:"wants"`
	Haves          int      `json:"haves"`
	SuggestedPrice *float64 `json:"suggested_price"`
	Genres         []string `json:"genres"`
	Styles         []string `json:"styles"`
	Year           *int     `json:"year"`
	Score          float64  `json:"score"`
}

type SellerKnapsack struct {
	Seller        string         `json:"seller"`
	Knapsack      []KnapsackItem `json:"knapsack"`
	Contenders    []KnapsackItem `json:"contenders"`
	TotalSelected int            `json:"total_selected"`
	TotalCost     float64        `json:"total_cost"`
	TotalScore    float64        `json:"total_score"`
}

type KnapsackRequest struct {
	Budget float64 `json:"budget"`
}

type KnapsackResponse struct {
	Knapsacks []SellerKnapsack `json:"knapsacks"`
	Budget    float64          `json:"budget"`
}

type KnapsackWeights struct {
	Embedding float64 `json:"embedding"`
	PriceDiff float64 `json:"price_diff"`
	IsWant    float64 `json:"is_want"`
}

type OptimizationResult struct {
	SelectedItems []Record `json:"selected_items"`
	TotalScore    float64  `json:"total_score"`
	TotalCost     float64  `json:"total_cost"`
	SellerName    string   `json:"seller_name"`
}

type ExchangeRateResponse struct {
	Rates map[string]float64 `json:"rates"`
}

type Bandit struct {
	ID            uint      `json:"id" gorm:"primaryKey"`
	Version       string    `json:"version" gorm:"not null"`
	ModelWeights  []byte    `json:"-" gorm:"type:bytea"`
	Hyperparams   string    `json:"hyperparams" gorm:"type:jsonb"`
	TrainingStats string    `json:"training_stats" gorm:"type:jsonb"`
	CreatedAt     time.Time `json:"created_at"`
	IsActive      bool      `json:"is_active" gorm:"default:false"`
}

type BanditTrainingInstance struct {
	ID        uint      `json:"id" gorm:"primaryKey"`
	RecordID  uint      `json:"record_id" gorm:"not null"`
	Record    Record    `json:"record" gorm:"foreignKey:RecordID"`
	Context   string    `json:"context" gorm:"type:jsonb"`
	Predicted bool      `json:"predicted"`
	Actual    bool      `json:"actual"`
	Reward    float64   `json:"reward"`
	Timestamp time.Time `json:"timestamp" gorm:"autoCreateTime"`
}

type BanditMetrics struct {
	ID            uint      `json:"id" gorm:"primaryKey"`
	ModelVersion  string    `json:"model_version" gorm:"not null"`
	Precision     float64   `json:"precision"`
	Recall        float64   `json:"recall"`
	F1Score       float64   `json:"f1_score"`
	ExploreRate   float64   `json:"explore_rate"`
	TotalExamples int       `json:"total_examples"`
	Timestamp     time.Time `json:"timestamp" gorm:"autoCreateTime"`
}

type BatchPerformance struct {
	ID        uint      `gorm:"primaryKey"`
	Timestamp time.Time `gorm:"autoCreateTime"`
	Correct   int       `json:"correct"`
	Total     int       `json:"total"`
	Accuracy  float64   `json:"accuracy"`
}

type EbayListing struct {
	ID           uint   `gorm:"primaryKey"`
	EbayID       string `gorm:"index:,unique;not null"`
	EbayTitle    string
	Price        string
	Currency     string
	CurrentBid   string
	BidCount     int
	EndDate      time.Time
	CreationDate time.Time

	// Enriched metadata (populated after save)
	Artist          string
	Album           string
	Label           string
	Format          string
	Year            string
	RecordCondition string
	SleeveCondition string
	Genre           string
	Style           string
	Keeper          bool `gorm:"default:false"`
}

type User struct {
	ID           uint      `json:"id"`
	Username     string    `json:"username"`
	PasswordHash string    `json:"-"`
	CreatedAt    time.Time `json:"created_at"`
}

func (Bandit) TableName() string {
	return "bandit_model"
}

func (BanditTrainingInstance) TableName() string {
	return "bandit_training_example"
}

func (BanditMetrics) TableName() string {
	return "bandit_metrics"
}

// TableName methods for custom table names to match Django
func (Record) TableName() string {
	return "discogs_discogsrecord"
}

func (DiscogsSeller) TableName() string {
	return "discogs_discogsseller"
}

func (DiscogsListing) TableName() string {
	return "discogs_discogslisting"
}

func (EbayListing) TableName() string {
	return "ebay_listings"
}

func (User) TableName() string {
	return "user"
}
