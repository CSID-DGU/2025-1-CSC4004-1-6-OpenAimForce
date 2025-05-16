package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

type InitRequest struct {
	Callback string `json:"callback"`
	APIKey   string `json:"api_key"`
}

// Read secret API key from config/hostpw
func isValidAPIKey(key string) bool {
	data, err := os.ReadFile("config/hostpw")
	if err != nil {
		log.Println("Failed to read hostpw:", err)
		return false
	}
	expected := strings.TrimSpace(string(data))
	return key == expected
}

// WebSocket dialer (distinct name to avoid collision)
var hostUpgrader = websocket.DefaultDialer

func GameServerInitHandler(w http.ResponseWriter, r *http.Request) {
	var req InitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
	if !isValidAPIKey(req.APIKey) {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	log.Println("Received handshake from:", req.Callback)

	// Dial the game server's WebSocket endpoint
	conn, _, err := hostUpgrader.Dial(req.Callback, nil)
	if err != nil {
		log.Printf("WebSocket dial failed: %v", err)
		http.Error(w, "WebSocket connect failed", http.StatusBadGateway)
		return
	}

	// Manage connection (example: periodic ping)
	go func() {
		defer conn.Close()
		for {
			err := conn.WriteMessage(websocket.TextMessage, []byte("PING"))
			if err != nil {
				log.Printf("WebSocket write error: %v", err)
				break
			}
			log.Println("Sent ping to game server")
			time.Sleep(10 * time.Second)
		}
	}()

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}
