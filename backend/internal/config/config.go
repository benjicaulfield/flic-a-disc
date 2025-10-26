package config

import (
	"os"
)

type Config struct {
	Database DatabaseConfig
	Server   ServerConfig
	External ExternalConfig
}

type DatabaseConfig struct {
	Host     string
	Port     string
	User     string
	Password string
	Name     string
	SSLMode  string
}

type ServerConfig struct {
	Port string
	Host string
}

type ExternalConfig struct {
	DiscogsConsumerKey    string
	DiscogsConsumerSecret string
	EbayAppId             string
	EbayDevId             string
	EbayCertId            string
}

func Load() *Config {
	return &Config{
		Database: DatabaseConfig{
			Host:     getEnv("DB_HOST", "localhost"),
			Port:     getEnv("DB_PORT", "5432"),
			User:     getEnv("DB_USER", ""),
			Password: getEnv("DB_PASSWORD", ""),
			Name:     getEnv("DB_NAME", ""),
			SSLMode:  getEnv("DB_SSLMODE", "disable"),
		},
		Server: ServerConfig{
			Port: getEnv("PORT", "8000"),
			Host: getEnv("HOST", "localhost"),
		},
		External: ExternalConfig{
			DiscogsConsumerKey:    getEnv("DISCOGS_CONSUMER_KEY", ""),
			DiscogsConsumerSecret: getEnv("DISCOGS_CONSUMER_SECRET", ""),
			EbayAppId:             getEnv("EBAY_APP_ID", ""),
			EbayDevId:             getEnv("EBAY_DEV_ID", ""),
			EbayCertId:            getEnv("EBAY_CERT_ID", ""),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
