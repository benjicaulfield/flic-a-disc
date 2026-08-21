// internal/handlers/rag.go
//
// RAGHandler is a thin reverse-proxy that forwards /api/rag/* requests to
// the Django service. We proxy instead of using httputil.ReverseProxy
// directly so that we can:
//
//   - stream Server-Sent-Events through gin without buffering
//   - preserve the path exactly (/api/rag/query/ → /api/rag/query/)
//   - keep the existing auth middleware in the call chain
package handlers

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

type RAGHandler struct {
	djangoURL string
	client    *http.Client
}

func NewRAGHandler() *RAGHandler {
	url := os.Getenv("DJANGO_URL")
	if url == "" {
		url = "http://localhost:8001"
	}
	url = strings.TrimRight(url, "/")

	return &RAGHandler{
		djangoURL: url,
		client: &http.Client{
			// generation with a local LLM can be slow; keep this generous
			Timeout: 15 * time.Minute,
			Transport: &http.Transport{
				DisableKeepAlives: true,
			},
		},
	}
}

// Proxy forwards the inbound request to <DJANGO_URL>/api/rag/<suffix> and
// copies the response back. It supports streaming bodies (SSE).
func (h *RAGHandler) Proxy(c *gin.Context) {
	suffix := c.Param("path") // leading slash included, e.g. "/query/"
	target := fmt.Sprintf("%s/api/rag%s", h.djangoURL, suffix)

	// Buffer the body so we can set Content-Length explicitly.
	// Django's dev server doesn't support chunked transfer encoding.
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read request body"})
		return
	}

	// Rebuild the upstream request.
	upstream, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		target,
		bytes.NewReader(body),
	)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to build upstream request"})
		return
	}
	upstream.ContentLength = int64(len(body))
	upstream.URL.RawQuery = c.Request.URL.RawQuery

	// Forward relevant headers. We strip hop-by-hop headers implicitly by
	// only copying what we need.
	for _, h := range []string{"Content-Type", "Accept"} {
		if v := c.GetHeader(h); v != "" {
			upstream.Header.Set(h, v)
		}
	}

	resp, err := h.client.Do(upstream)
	if err != nil {
		log.Printf("rag proxy error → %s: %v", target, err)
		c.JSON(http.StatusBadGateway, gin.H{"error": "django unreachable", "detail": err.Error()})
		return
	}
	defer resp.Body.Close()

	// Mirror upstream headers to the client.
	for k, vv := range resp.Header {
		for _, v := range vv {
			c.Writer.Header().Add(k, v)
		}
	}
	c.Writer.WriteHeader(resp.StatusCode)

	ct := resp.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "text/event-stream") {
		streamSSE(c, resp.Body)
		return
	}

	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		log.Printf("rag proxy copy error: %v", err)
	}
}

// streamSSE copies an SSE response, flushing after each line so the browser
// receives tokens as soon as Django emits them.
func streamSSE(c *gin.Context, body io.Reader) {
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		// fall back to a blocking copy
		io.Copy(c.Writer, body)
		return
	}

	reader := bufio.NewReader(body)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if _, werr := c.Writer.Write(line); werr != nil {
				return
			}
			flusher.Flush()
		}
		if err != nil {
			return
		}
	}
}
