// go-backend/cmd/api/main.go
package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"flic-a-disc/internal/config"
	"flic-a-disc/internal/database"
	"flic-a-disc/internal/handlers"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	// Load .env from module root; non-fatal if missing
	_ = godotenv.Load(".env")

	cfg := config.Load()

	db, err := database.Initialize(cfg.Database)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	if err := database.AutoMigrate(db); err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
	}

	// Gin setup
	gin.SetMode(gin.ReleaseMode) // or gin.DebugMode during dev
	r := gin.New()
	r.Use(
		cors.New(cors.Config{
			AllowOrigins:     []string{"http://localhost:3000", "http://localhost:5173", "http://localhost:5174"},
			AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
			AllowHeaders:     []string{"Origin", "Content-Length", "Content-Type", "Authorization", "X-Requested-With", "X-CSRF-Token"},
			AllowCredentials: true,
			MaxAge:           12 * time.Hour,
		}),
	)

	h := handlers.New(db, cfg)
	ebayHandler := handlers.NewEbayHandler(cfg.External.EbayAppId, cfg.External.EbayCertId, db)
	authHandler := handlers.NewAuthHandler(db)

	r.POST("/api/auth/login", authHandler.Login)
	r.POST("/api/auth/logout", authHandler.Logout)

	protected := r.Group("/api")
	protected.Use(authHandler.AuthMiddleware())
	{
		protected.GET("/discogs/keepers", h.GetDiscogsKeepersPage)
		protected.GET("/discogs/stats", h.GetStats)
		protected.POST("/discogs/labels", h.LabelRecords)
		protected.GET("/discogs/wanted", h.GetWantedRecords)
		protected.POST("/discogs/performance", h.RecordBatchPerformance)
		protected.GET("/ebay/auctions", ebayHandler.TriggerFetch)
		protected.GET("/discogs/select_batch", h.GetDiscogsKeepersPage)
		protected.GET("/ebay/recommend", ebayHandler.RecommendEbayListings)
		protected.GET("/ebay/unannotated", ebayHandler.GetUnannotatedListings)
		protected.GET("auth/me", authHandler.Me)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("API listening on :%s", port)
	log.Fatal(srv.ListenAndServe())
}
