package handlers

import (
	"database/sql"
	"log"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
)

const (
	thresholdTime1 = 10 // seconds
	thresholdTime2 = 20
)

type QueueConn struct {
	PID      int
	MMR      int
	JoinedAt time.Time
	WS       *websocket.Conn
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

var (
	queue      []QueueConn
	queueMutex sync.Mutex
	dbHandle   *sql.DB
)

func StartQueueManager(matchCount int) {
	go func() {
		for {
			time.Sleep(2 * time.Second)

			queueMutex.Lock()
			candidates := make([]QueueConn, len(queue))
			copy(candidates, queue)
			now := time.Now()

			var matchedGroup []QueueConn
			matchIndex := -1

			// Try every group of matchCount
			for i := 0; i+matchCount <= len(candidates); i++ {
				group := candidates[i : i+matchCount]
				baseMMR := group[0].MMR
				maxMMR := baseMMR
				minMMR := baseMMR

				for _, c := range group {
					if c.MMR > maxMMR {
						maxMMR = c.MMR
					}
					if c.MMR < minMMR {
						minMMR = c.MMR
					}
				}

				span := maxMMR - minMMR
				wait := int(now.Sub(group[0].JoinedAt).Seconds())

				allowed := span == 0 || (wait >= thresholdTime1 && span <= 1) || (wait >= thresholdTime2 && span <= 2)
				if allowed {
					matchedGroup = group
					matchIndex = i
					break
				}
			}

			if matchIndex != -1 {
				queue = append(queue[:matchIndex], queue[matchIndex+matchCount:]...)
			}
			queueMutex.Unlock()

			if matchIndex == -1 {
				continue
			}

			// Split into two teams by alternating sorted MMRs
			teamA := []QueueConn{}
			teamB := []QueueConn{}
			sorted := matchedGroup
			for i := 0; i < len(sorted); i++ {
				if i%2 == 0 {
					teamA = append(teamA, sorted[i])
				} else {
					teamB = append(teamB, sorted[i])
				}
			}

			rand.Shuffle(len(teamA), func(i, j int) { teamA[i], teamA[j] = teamA[j], teamA[i] })
			rand.Shuffle(len(teamB), func(i, j int) { teamB[i], teamB[j] = teamB[j], teamB[i] })

			msg := func(aimhack bool) string {
				if aimhack {
					return `CON/221.139.184.184/28763/1234/1/1`
				}
				return `CON/221.139.184.184/28763/1234/1/1`
			}

			for i, c := range append(teamA, teamB...) {
				isAim := (i == 0 || i == len(teamA)) // one per team
				c.WS.WriteMessage(websocket.TextMessage, []byte(msg(isAim)))
				c.WS.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "matched"))
				c.WS.Close()
			}

			log.Printf("Matched %d players", matchCount)
		}
	}()
}

func QueueHandler(secret string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		tokenStr := strings.TrimPrefix(auth, "Bearer ")

		tok, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
			return []byte(secret), nil
		})
		if err != nil || !tok.Valid {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		claims, ok := tok.Claims.(jwt.MapClaims)
		if !ok || claims["pid"] == nil {
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}
		pid := int(claims["pid"].(float64))

		// 🔽 Get MMR from DB
		var mmr int
		err = dbHandle.QueryRow("CALL get_mmr(?)", pid).Scan(&mmr)
		if err != nil {
			http.Error(w, "mmr lookup failed", http.StatusInternalServerError)
			return
		}

		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("WebSocket upgrade failed:", err)
			return
		}

		conn := QueueConn{PID: pid, MMR: mmr, JoinedAt: time.Now(), WS: ws}
		queueMutex.Lock()
		queue = append(queue, conn)
		queueMutex.Unlock()

		log.Printf("Player %d (MMR %d) joined queue", pid, mmr)

		for {
			msgType, msg, err := ws.ReadMessage()
			if err != nil || (msgType == websocket.TextMessage && string(msg) == "exit") {
				break
			}
		}

		queueMutex.Lock()
		newQueue := queue[:0]
		for _, c := range queue {
			if c.WS != ws {
				newQueue = append(newQueue, c)
			}
		}
		queue = newQueue
		queueMutex.Unlock()

		ws.Close()
		log.Printf("Player %d left queue", pid)
	}
}

// Store DB handle once
func SetDB(db *sql.DB) {
	dbHandle = db
}
