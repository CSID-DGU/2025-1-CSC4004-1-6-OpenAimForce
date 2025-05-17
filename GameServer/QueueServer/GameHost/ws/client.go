package ws

import (
	"GameHost/bridge"
	"GameHost/procedures"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type GameHostClient struct {
	conn     *websocket.Conn
	connMu   sync.Mutex
	hostMgr  *procedures.HostManager
	runPath  string
	ip       string
	masterWS string
}

var globalClient *GameHostClient

func InitClient(hostMgr *procedures.HostManager, runPath, ip, masterURL string) {
	globalClient = &GameHostClient{
		hostMgr:  hostMgr,
		runPath:  runPath,
		ip:       ip,
		masterWS: masterURL,
	}
	go globalClient.loopConnect()
	bridge.RegisterSender(globalClient.Send)
}

func (c *GameHostClient) loopConnect() {
	for {
		conn, _, err := websocket.DefaultDialer.Dial(c.masterWS, nil)
		if err != nil {
			log.Println("Failed to connect to master:", err)
			time.Sleep(5 * time.Second)
			continue
		}
		log.Println("Connected to master:", c.masterWS)

		c.connMu.Lock()
		c.conn = conn
		c.connMu.Unlock()

		c.handleConnection(conn)
	}
}

func (c *GameHostClient) handleConnection(conn *websocket.Conn) {
	defer conn.Close()
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("WebSocket read error:", err)
			break
		}

		cmd := string(msg)
		if cmd == "PING" {
			log.Println("Received ping from master")
			continue
		}

		fields := strings.Fields(cmd)
		if len(fields) == 0 {
			c.Send("ERR invalid command")
			continue
		}

		switch fields[0] {
		case "CREATE":
			if len(fields) != 4 {
				c.Send("ERR usage: CREATE <tempId> <team1> <team2>")
				continue
			}
			tempId := fields[1]
			team1 := fields[2]
			team2 := fields[3]
			resp := c.hostMgr.CreateGame(c.ip, c.runPath, team1, team2)
			if resp == "FAIL" {
				c.Send("FAIL " + tempId)
			} else {
				c.Send("SUCCESS " + tempId + " " + resp)
			}

		case "KILL":
			if len(fields) != 2 {
				c.Send("ERR usage: KILL <pwport>")
				continue
			}
			pwport := fields[1]
			ok := c.hostMgr.TerminateGameByPwPort(pwport)
			if ok {
				c.Send("KILL_OK " + pwport)
			} else {
				c.Send("KILL_FAIL " + pwport)
			}

		default:
			c.Send("ERR unknown command")
		}
	}
}

func (c *GameHostClient) Send(msg string) {
	c.connMu.Lock()
	defer c.connMu.Unlock()
	if c.conn != nil {
		err := c.conn.WriteMessage(websocket.TextMessage, []byte(msg))
		if err != nil {
			log.Println("Send error:", err)
		}
	}
}
