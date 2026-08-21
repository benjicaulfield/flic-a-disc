package handlers

import (
	"github.com/benjicaulfield/flic-a-disc/internal/models"
	"github.com/gin-gonic/gin"
)

func (h *Handler) GetTodos(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	var todos []models.Todo
	if err := h.DB.Where("user_id = ?", userID).Order("\"order\" asc").Find(&todos).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch todos"})
		return
	}

	c.JSON(200, todos)
}

func (h *Handler) GetRecentCompletedTodos(c *gin.Context) {
	var todos []models.Todo
	if err := h.DB.Where("status = ?", "done").Order("updated_at desc").Limit(5).Find(&todos).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch recent completed todos"})
		return
	}

	c.JSON(200, todos)
}

func (h *Handler) CreateTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	var body struct {
		Text   string `json:"text"`
		Status string `json:"status"`
		Order  int    `json:"order"`
	}
	if err := c.BindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": "Invalid body"})
		return
	}

	if body.Status == "" {
		body.Status = "backlog"
	}

	todo := models.Todo{
		UserID: userID.(uint),
		Text:   body.Text,
		Status: body.Status,
		Order:  body.Order,
	}

	if err := h.DB.Create(&todo).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to create todo"})
		return
	}

	c.JSON(201, todo)
}

func (h *Handler) UpdateTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	todoID := c.Param("id")

	var body struct {
		Text   *string `json:"text"`
		Status *string `json:"status"`
		Order  *int    `json:"order"`
	}
	if err := c.BindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": "Invalid body"})
		return
	}

	var todo models.Todo
	if err := h.DB.Where("id = ? AND user_id = ?", todoID, userID).First(&todo).Error; err != nil {
		c.JSON(404, gin.H{"error": "Not found"})
		return
	}

	updates := map[string]interface{}{}
	if body.Text != nil {
		updates["text"] = *body.Text
	}
	if body.Status != nil {
		updates["status"] = *body.Status
	}
	if body.Order != nil {
		updates["order"] = *body.Order
	}

	if err := h.DB.Model(&todo).Updates(updates).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to update todo"})
		return
	}

	c.JSON(200, todo)
}

func (h *Handler) DeleteTodo(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "Unauthorized"})
		return
	}

	todoID := c.Param("id")

	if err := h.DB.Where("id = ? AND user_id = ?", todoID, userID).Delete(&models.Todo{}).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to delete todo"})
		return
	}

	c.Status(204)
}
