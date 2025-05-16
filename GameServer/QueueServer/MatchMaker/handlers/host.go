package handlers

import (
	"database/sql"
	"errors"
	"log"
	"matchmaker/models"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type GameHostConn struct {
	conn   *websocket.Conn
	connMu sync.Mutex
	addr   string
}

var hostUpgrader = websocket.Upgrader{}
var currentHost *GameHostConn
var hostDB *sql.DB

// For synchronous command-reply:
var (
	createReplyChans   = make(map[string]chan [3]string) // [ip, port, pw]
	createReplyChansMu sync.Mutex
)

func HandleIncomingGameHost(w http.ResponseWriter, r *http.Request) {
	conn, err := hostUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	log.Println("GameHost connected:", r.RemoteAddr)
	host := &GameHostConn{conn: conn, addr: r.RemoteAddr}
	currentHost = host

	go readLoop(conn)
	go pingLoop(conn)
}

func pingLoop(conn *websocket.Conn) {
	for {
		currentHost.connMu.Lock()
		err := conn.WriteMessage(websocket.TextMessage, []byte("PING"))
		currentHost.connMu.Unlock()
		if err != nil {
			log.Println("Write error:", err)
			break
		}
		log.Println("Sent ping to GameHost")
		time.Sleep(10 * time.Second)
	}
}

func readLoop(conn *websocket.Conn) {
	defer conn.Close()
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("Read error from GameHost:", err)
			break
		}
		text := string(msg)
		log.Println("From GameHost:", text)

		tokens := strings.Fields(text)
		if len(tokens) == 0 {
			continue
		}

		switch tokens[0] {
		case "KILL_DUE_TO_TIMEOUT":
			if len(tokens) >= 2 {
				log.Println("[TIMEOUT] Game killed:", tokens[1])
			}
		case "KILL_OK":
			if len(tokens) >= 2 {
				log.Println("[KILLED] Game ended by request:", tokens[1])
			}
		case "KILL_FAIL":
			if len(tokens) >= 2 {
				log.Println("[FAIL] Kill failed:", tokens[1])
			}
		case "SUCCESS":
			// SUCCESS <tempId> <ip> <port> <pw>
			if len(tokens) >= 5 {
				tempId := tokens[1]
				ip := tokens[2]
				port := tokens[3]
				pw := tokens[4]
				notifyCreateReply(tempId, ip, port, pw)
				log.Printf("[CREATE SUCCESS] %s → %s / %s / %s\n", tempId, ip, port, pw)
			}
		case "FAIL":
			if len(tokens) >= 2 {
				tempId := tokens[1]
				notifyCreateReply(tempId, "", "", "")
				log.Println("[CREATE FAIL]", tempId)
			}
		case "GAMEEND":
			// GAMEEND <pwport> <result>/<ingameID> <kill> <death> <quit> <team>/...
			if len(tokens) >= 3 {
				pwport := tokens[1]

				// Reconstruct the full player entry string (since entries may have spaces and be split across tokens)
				entryString := strings.Join(tokens[2:], " ")
				log.Printf("[GAMEEND] Raw entry string: %q", entryString)

				// Split at "/" (first part is result, rest are player entries)
				all := strings.Split(entryString, "/")
				if len(all) < 2 {
					log.Printf("[GAMEEND] Malformed: after split by '/', got less than 2 fields: %#v", all)
					continue
				}

				result := all[0]
				entries := all[1:] // e.g. [ "evan223 0 0 1 team1", "inchan24223 0 0 0 team2" ]
				log.Printf("[GAMEEND] pwport=%s, result=%s, %d player entries", pwport, result, len(entries))
				for i, e := range entries {
					log.Printf("[GAMEEND] Entry %d: %q", i, e)
				}

				models.HandleGameEnd(pwport, result, entries)
			} else {
				log.Printf("[GAMEEND] Too few tokens: %#v", tokens)
			}
		}
	}
}

func SendToGameHost(msg string) {
	if currentHost == nil {
		log.Println("No GameHost connected")
		return
	}
	currentHost.connMu.Lock()
	defer currentHost.connMu.Unlock()
	err := currentHost.conn.WriteMessage(websocket.TextMessage, []byte(msg))
	if err != nil {
		log.Println("Failed to send message:", err)
	}
}

// Wait for a "SUCCESS tempId ip pwport" from GameHost
func WaitForCreateReply(tempId string, timeout time.Duration) (string, string, string, error) {
	createReplyChansMu.Lock()
	ch, ok := createReplyChans[tempId]
	if !ok {
		ch = make(chan [3]string, 1)
		createReplyChans[tempId] = ch
	}
	createReplyChansMu.Unlock()

	select {
	case arr := <-ch:
		ip := arr[0]
		port := arr[1]
		pw := arr[2]
		if ip == "" || port == "" || pw == "" {
			return "", "", "", errors.New("GameHost failed to create game")
		}
		return ip, port, pw, nil
	case <-time.After(timeout):
		return "", "", "", errors.New("timeout waiting for GameHost")
	}
}

func notifyCreateReply(tempId, ip, port, pw string) {
	createReplyChansMu.Lock()
	ch, ok := createReplyChans[tempId]
	if ok {
		ch <- [3]string{ip, port, pw}
		close(ch)
		delete(createReplyChans, tempId)
	}
	createReplyChansMu.Unlock()
}

// Store DB handle once
func SetDBHost(db *sql.DB) {
	hostDB = db
}

// Hack to avoid cyclic import for models.
func requireModels() *struct {
	ConfirmGame   func(string)
	HandleGameEnd func(string, string, []string)
} {
	panic("please replace with DI or refactor")
}
