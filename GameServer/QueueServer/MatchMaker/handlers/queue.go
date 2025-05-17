package handlers

import (
	"database/sql"
	"fmt"
	"log"
	"matchmaker/models"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
)

const (
	thresholdTime1 = 60 // seconds
	thresholdTime2 = 120
)

var allowedPeriods [][2]time.Time

type QueueConn struct {
	PID      int
	MMR      int
	ID       string
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
	tid        atomic.Uint64
)

func SetAllowedPeriods(periods [][2]time.Time) {
	allowedPeriods = periods
}

func isWithinAllowedPeriod(now time.Time) bool {
	for _, rng := range allowedPeriods {
		if now.Equal(rng[0]) || now.Equal(rng[1]) || (now.After(rng[0]) && now.Before(rng[1])) {
			return true
		}
	}
	return false
}

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

			// --- Create new game ---
			tempId := fmt.Sprintf("G%v", time.Now().UnixNano())
			teamAids := make([]string, len(teamA))
			teamBids := make([]string, len(teamB))
			for i, c := range teamA {
				teamAids[i] = c.ID
			}
			for i, c := range teamB {
				teamBids[i] = c.ID
			}

			// Send CREATE and wait for host response
			ip, port, pw, err := models.StartNewGameSync(
				tempId,
				strings.Join(teamAids, ","),
				strings.Join(teamBids, ","),
				SendToGameHost,
				WaitForCreateReply,
			)
			if err != nil {
				log.Printf("Failed to create game: %v", err)
				continue
			}

			// ---- Send connection info to clients ----
			aIdx := rand.Intn(len(teamA))
			bIdx := rand.Intn(len(teamB))
			espA := rand.Intn(2)         // 0 or 1
			espB := rand.Intn(2)         // 0 or 1
			aimhackA := 2 + rand.Intn(3) // 2, 3, or 4
			aimhackB := 2 + rand.Intn(3) // 2, 3, or 4

			for i, c := range append(teamA, teamB...) {
				esp, aimhack := 0, 1
				if i < len(teamA) && i == aIdx {
					esp = espA
					aimhack = aimhackA
				}
				if i >= len(teamA) && (i-len(teamA)) == bIdx {
					esp = espB
					aimhack = aimhackB
				}
				msg := fmt.Sprintf("CON/%s/%s/%s/%d/%d", ip, port, pw, esp, aimhack)
				log.Println("Send msg: " + msg)
				c.WS.WriteMessage(websocket.TextMessage, []byte(msg))
				c.WS.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "matched"))
				c.WS.Close()
			}
			log.Printf("Matched %d players", matchCount)

			// ---- TRACK GAME FOR HACK STATUS AND LOGFILE ----
			players := make([]models.PlayerHackStatus, 0, len(teamA)+len(teamB))

			for i, c := range teamA {
				esp := false
				aimhack := 1
				if i == aIdx {
					esp = (espA == 1)
					aimhack = aimhackA
				}
				players = append(players, models.PlayerHackStatus{
					PID:      c.PID,
					IngameID: c.ID,
					Team:     "team1",
					ESP:      esp,
					Aimhack:  aimhack,
				})
			}
			for i, c := range teamB {
				esp := false
				aimhack := 1
				if i == bIdx {
					esp = (espB == 1)
					aimhack = aimhackB
				}
				players = append(players, models.PlayerHackStatus{
					PID:      c.PID,
					IngameID: c.ID,
					Team:     "team2",
					ESP:      esp,
					Aimhack:  aimhack,
				})
			}
			logfile := fmt.Sprintf("%s_%s.log", pw, port)
			models.AddTrackedGame(pw, port, logfile, players)
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

		// Get MMR from DB
		var mmr int
		err = dbHandle.QueryRow("CALL get_mmr(?)", pid).Scan(&mmr)
		if err != nil {
			http.Error(w, "mmr lookup failed", http.StatusInternalServerError)
			return
		}

		// Get ingame_id and account_type from db
		var id, accountType string
		err = dbHandle.QueryRow("SELECT ingame_id, account_type FROM Player WHERE pid = ?", pid).Scan(&id, &accountType)
		if err != nil {
			http.Error(w, "lookup failed", http.StatusInternalServerError)
			return
		}

		// 셧다운제 (time limit)
		if accountType != "admin" && !isWithinAllowedPeriod(time.Now()) {
			http.Error(w, "queueing not allowed at this time", http.StatusForbidden)
			return
		}

		// Prevent same player from entering queue at a given moment
		queueMutex.Lock()
		for _, c := range queue {
			if c.PID == pid {
				queueMutex.Unlock()
				http.Error(w, "already in queue", http.StatusConflict)
				return
			}
		}
		queueMutex.Unlock()

		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("WebSocket upgrade failed:", err)
			return
		}

		conn := QueueConn{PID: pid, MMR: mmr, ID: id, JoinedAt: time.Now(), WS: ws}
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
func SetDBQueue(db *sql.DB) {
	dbHandle = db
}
