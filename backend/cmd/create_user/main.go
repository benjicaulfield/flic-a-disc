package main

import (
	"fmt"
	"log"

	"flic-a-disc/internal/config"
	"flic-a-disc/internal/database"
	"flic-a-disc/internal/models"

	"github.com/joho/godotenv"
	"golang.org/x/crypto/bcrypt"
)

func main() {
	// Load .env
	_ = godotenv.Load(".env")

	cfg := config.Load()

	db, err := database.Initialize(cfg.Database)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	username := "benjicaulfield"
	password := "grahamlambkin"

	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword(
		[]byte(password),
		bcrypt.DefaultCost,
	)
	if err != nil {
		log.Fatalf("Failed to hash password: %v", err)
	}

	// Create user
	user := models.User{
		Username:     username,
		PasswordHash: string(hashedPassword),
	}

	result := db.Create(&user)
	if result.Error != nil {
		log.Fatalf("Failed to create user: %v", result.Error)
	}

	fmt.Println("✅ User created successfully!")
	fmt.Printf("Username: %s\n", username)
	fmt.Printf("User ID: %d\n", user.ID)
}
