package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true }, // allow all for demo
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
	html := `
	<!DOCTYPE html>
	<html>
	<head><title>Matchmaker Demo</title></head>
	<body>
		<h1>It works!</h1>
		<script>
			const ws = new WebSocket("wss://" + location.host + "/ws");
			ws.onopen = () => ws.send("hello from browser");
			ws.onmessage = e => console.log("WS received:", e.data);
		</script>
	</body>
	</html>`
	fmt.Fprint(w, html)
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade failed:", err)
		return
	}
	defer conn.Close()

	log.Println("WebSocket connected")
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("Read error:", err)
			break
		}
		log.Println("Received:", string(msg))
		conn.WriteMessage(websocket.TextMessage, []byte("echo: "+string(msg)))
	}
}

func main() {
	http.HandleFunc("/", homeHandler)
	http.HandleFunc("/ws", wsHandler)

	log.Println("Listening on :8080 (behind nginx)")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal("Server error:", err)
	}
}
