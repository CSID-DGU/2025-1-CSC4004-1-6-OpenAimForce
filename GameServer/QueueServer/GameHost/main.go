package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/gorilla/websocket"
)

// --- Structs ---

type InitRequest struct {
	Callback string `json:"callback"`
	APIKey   string `json:"api_key"`
}

// --- WebSocket Handler ---

var upgrader = websocket.Upgrader{}

func WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer conn.Close()
	log.Println("WebSocket connection established")

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("WebSocket read error:", err)
			break
		}
		log.Println("Received:", string(msg))
	}
}

// --- Helper: Read Secret File ---

func readSecret(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("Failed to read %s: %v", path, err)
	}
	return strings.TrimSpace(string(data))
}

// --- Initiate Handshake with Remote Server ---

func initRemoteHandshake() {
	url := readSecret("secrets/url")
	key := readSecret("secrets/key")
	myCallback := "wss://your-domain.com/ws"

	body := InitRequest{
		Callback: myCallback,
		APIKey:   key,
	}
	jsonData, err := json.Marshal(body)
	if err != nil {
		log.Fatalf("Failed to marshal request: %v", err)
	}

	resp, err := http.Post(url+"/api/gs", "application/json", bytes.NewReader(jsonData))
	if err != nil {
		log.Fatalf("Failed to contact target server: %v", err)
	}
	defer resp.Body.Close()

	log.Println("Handshake response status:", resp.Status)
}

// --- Main ---

func main() {
	runPath := readSecret("secrets/run")
	http.HandleFunc("/ws", WebSocketHandler)
	http.Handle("/", http.FileServer(http.Dir("static")))

	go initRemoteHandshake()

	log.Println("Server started on :24999")
	log.Fatal(http.ListenAndServe(":24999", nil))
}
