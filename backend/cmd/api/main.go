// go-backend/cmd/api/main.go
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/benjicaulfield/flic-a-disc/internal/config"
	"github.com/benjicaulfield/flic-a-disc/internal/database"
	"github.com/benjicaulfield/flic-a-disc/internal/handlers"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(".env.local"); err == nil {
		fmt.Println("Loaded environment from .env.local")
	} else if err := godotenv.Load(".env"); err == nil {
		fmt.Println("Loaded environment from .env")
	} else {
		fmt.Println("No .env file found — using system environment")
	}

	cfg := config.Load()

	fmt.Printf("DB → host=%s user=%s db=%s sslmode=%s\n",
		cfg.Database.Host, cfg.Database.User, cfg.Database.Name, cfg.Database.SSLMode)

	db, err := database.Initialize(cfg.Database)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	if err := database.AutoMigrate(db); err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
	}

	// Gin setup
	gin.SetMode(gin.ReleaseMode) // or gin.DebugMode during dev
	r := gin.Default()
	r.Use(cors.New(cors.Config{
		AllowOrigins: []string{
			"https://flic-a-disc.com",
			"https://www.flic-a-disc.com",
			"http://localhost:5173",
			"http://localhost:5174",
			"http://localhost:3000",
		},
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Length", "Content-Type", "Authorization"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))
	r.OPTIONS("/*path", func(c *gin.Context) {
		c.Status(200)
	})

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
		protected.GET("/ebay/auctions", ebayHandler.GetListings)
		protected.POST("/ebay/refresh", ebayHandler.TriggerFetch)
		protected.GET("/discogs/select_batch", h.GetDiscogsKeepersPage)
		protected.GET("/ebay/recommend", ebayHandler.RecommendEbayListings)
		protected.GET("/ebay/unannotated", ebayHandler.GetUnannotatedListings)
		protected.GET("/ebay/save_keepers", ebayHandler.SaveKeepers)
		protected.GET("/auth/me", authHandler.Me)
		protected.GET("/todos", h.GetTodos)
		protected.POST("/todos", h.CreateTodo)
		protected.PATCH("/todos/:id", h.UpdateTodo)
		protected.DELETE("/todos/:id", h.DeleteTodo)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
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
