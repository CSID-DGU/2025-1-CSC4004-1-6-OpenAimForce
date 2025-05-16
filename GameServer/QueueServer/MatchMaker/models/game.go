package models

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"
)

const trackedGameExpiry = 10 * time.Minute // games older than this are removed

type PlayerHackStatus struct {
	PID      int // Optional: Can also use string ID if needed
	IngameID string
	Team     string
	ESP      bool
	Aimhack  int
}

type GameTracker struct {
	Pw      string
	Port    string
	Logfile string
	Players []PlayerHackStatus
	Time    time.Time
}

var (
	trackedGames   []GameTracker
	trackedGamesMu sync.Mutex
)

type GameInstance struct {
	IP        string
	TempID    string
	PwPort    string
	StartTime time.Time
	Confirmed bool
}

var db *sql.DB

var (
	GameMap   = make(map[string]*GameInstance)
	GameMapMu sync.Mutex
)

func SetDB(mdb *sql.DB) {
	db = mdb
}

func NewGame(tempId string) {
	GameMapMu.Lock()
	defer GameMapMu.Unlock()
	GameMap[tempId] = &GameInstance{
		TempID:    tempId,
		StartTime: time.Now(),
	}
}

// Synchronous start: send to GameHost, wait for reply, return pwport or error/timeout
func StartNewGameSync(tempId, team1, team2 string, send func(cmd string), waitReply func(tempId string, timeout time.Duration) (string, string, string, error)) (string, string, string, error) {
	NewGame(tempId)
	cmd := "CREATE " + tempId + " " + team1 + " " + team2
	send(cmd)
	ip, port, pw, err := waitReply(tempId, 10*time.Second)
	if err != nil {
		return "", "", "", err
	}
	GameMapMu.Lock()
	defer GameMapMu.Unlock()
	GameMap[tempId].IP = ip
	GameMap[tempId].PwPort = port // Could rename to Port
	// Store pw in struct if you want; else skip
	return ip, port, pw, nil
}

func ConfirmGame(pwport string) {
	GameMapMu.Lock()
	defer GameMapMu.Unlock()
	for _, g := range GameMap {
		if g.PwPort == pwport {
			g.Confirmed = true
			fmt.Println("Join confirmed for", pwport)
		}
	}
}

/*
GAMEEND protocol format (received from GameHost):

GAMEEND <pwport> <result>/
  <ingameID> <kill> <death> <quit> <team>/
  <ingameID> <kill> <death> <quit> <team>/...

- <pwport>: password_port string, e.g. Mt4fDN82UemBnhbMKtc42evtz_25000
- <result>: "1", "2", or "draw"
- Each player entry: <ingameID> <kill> <death> <quit> <team>
    - <team>: "team1" or "team2"
Entries are '/'-separated after <result>
*/

func HandleGameEnd(pwport string, result string, entries []string) {
	// Extract password part (pw), if pwport is password_port format
	pw := pwport
	if strings.Contains(pwport, "_") {
		split := strings.SplitN(pwport, "_", 2)
		pw = split[0]
	}

	var (
		logfile      string
		foundPlayers []PlayerHackStatus
		//foundEntry   *GameTracker
	)
	trackedGamesMu.Lock()
	found := false
	nowTime := time.Now()
	newTrackedGames := trackedGames[:0]

	for idx, g := range trackedGames {
		fmt.Printf("[HandleGameEnd][tracking] Checking trackedGames[%d]: Pw=%q, Logfile=%q, Port=%q, Players=%d, Time=%v\n",
			idx, g.Pw, g.Logfile, g.Port, len(g.Players), g.Time)
		// Remove expired games
		if nowTime.Sub(g.Time) > trackedGameExpiry {
			fmt.Printf("[HandleGameEnd][tracking] Skipping expired game: Pw=%q (expired by %v)\n", g.Pw, nowTime.Sub(g.Time))
			continue
		}
		// Match by pw prefix only for foundPlayers/logfile
		if !found && strings.HasPrefix(g.Pw, pw) {
			foundPlayers = g.Players
			logfile = g.Logfile
			//foundEntry = &g
			found = true
			fmt.Printf("[HandleGameEnd][tracking] Found tracked entry for pwport prefix: Pw=%q (logfile=%q, players=%d)\n", g.Pw, g.Logfile, len(g.Players))
			// Do not append, i.e., remove this matched game from newTrackedGames
			continue
		}
		newTrackedGames = append(newTrackedGames, g)
	}
	trackedGames = newTrackedGames
	trackedGamesMu.Unlock()

	if !found {
		logfile = "" // NULL in DB
		fmt.Printf("[HandleGameEnd] No tracked entry found for pwport prefix=%q; logfile set to empty\n", pw)
	} else {
		fmt.Printf("[HandleGameEnd] Matched tracked entry: logfile=%q, foundPlayers=%v\n", logfile, foundPlayers)
	}

	// Determine winner field
	winner := "draw"
	if result == "1" {
		winner = "team1"
	} else if result == "2" {
		winner = "team2"
	}
	fmt.Printf("[HandleGameEnd] pwport=%s, winner=%s (%q), entries=%d, logfile=%q\n", pwport, winner, result, len(entries), logfile)

	now := time.Now()
	tx, err := db.Begin()
	if err != nil {
		fmt.Printf("[HandleGameEnd] Failed to start transaction: %v\n", err)
		return
	}

	// 1. Insert into Game
	res, err := tx.Exec("INSERT INTO Game (winner, game_time, logfile_name) VALUES (?, ?, ?)", winner, now, logfile)
	if err != nil {
		fmt.Printf("[HandleGameEnd] Failed to insert into Game: %v\n", err)
		_ = tx.Rollback()
		return
	}
	gameID, err := res.LastInsertId()
	if err != nil {
		fmt.Printf("[HandleGameEnd] Failed to get game_id: %v\n", err)
		_ = tx.Rollback()
		return
	}
	fmt.Printf("[HandleGameEnd] Inserted Game (id=%d, winner=%s, logfile=%q)\n", gameID, winner, logfile)

	// 2. Insert each player entry
	for idx, entry := range entries {
		fmt.Printf("[HandleGameEnd] Parsing entry %d: %q\n", idx, entry)
		tokens := strings.Fields(entry)
		fmt.Printf("[HandleGameEnd] Tokens: %#v\n", tokens)
		if len(tokens) != 5 { // <ingameID> <kill> <death> <quit> <team>
			fmt.Printf("[HandleGameEnd] Malformed player entry (need 5 fields): %q\n", entry)
			continue
		}
		ingameID := tokens[0]
		kills := tokens[1]
		deaths := tokens[2]
		quit := tokens[3]
		teamVal := tokens[4]

		// Find pid for ingame_id
		var pid int
		err = tx.QueryRow("SELECT pid FROM Player WHERE ingame_id = ?", ingameID).Scan(&pid)
		if err != nil {
			fmt.Printf("[HandleGameEnd] Player not found (ingame_id=%q): %v\n", ingameID, err)
			pid = -1
		} else {
			fmt.Printf("[HandleGameEnd] Player found (ingame_id=%q): pid=%d\n", ingameID, pid)
		}

		var quitTime interface{}
		if quit == "1" {
			quitTime = now
			fmt.Printf("[HandleGameEnd] Player %s quit; quitTime will be set\n", ingameID)
		} else {
			quitTime = nil
		}

		esp := -1
		aimhack := -1
		origTeam := teamVal
		// Try to find player in foundPlayers
		foundHack := false
		for j, ph := range foundPlayers {
			matchPid := (pid != -1 && ph.PID == pid)
			matchName := (ph.IngameID == ingameID)
			if matchPid || matchName {
				foundHack = true
				fmt.Printf("[HandleGameEnd] Found player in tracked: index=%d, PID=%d, IngameID=%q, Team=%q, ESP=%v, Aimhack=%d\n",
					j, ph.PID, ph.IngameID, ph.Team, ph.ESP, ph.Aimhack)
				if ph.ESP {
					esp = 1
				} else {
					esp = 0
				}
				aimhack = ph.Aimhack
				teamVal = ph.Team
				break
			}
		}
		if !foundHack {
			fmt.Printf("[HandleGameEnd] No tracked hack/esp/team for player %q; using incoming: esp=%d, aimhack=%d, team=%q\n", ingameID, esp, aimhack, origTeam)
		}

		_, err = tx.Exec(
			"INSERT INTO GameParticipation (game_id, pid, team, kills, deaths, quitTime, esp, aimhack) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
			gameID, pid, teamVal, kills, deaths, quitTime, esp, aimhack,
		)
		if err != nil {
			fmt.Printf("[HandleGameEnd] Failed to insert participation for %q: %v\n", ingameID, err)
		} else {
			fmt.Printf("[HandleGameEnd] Participation row inserted for %q (pid=%d, team=%q, esp=%d, aimhack=%d)\n", ingameID, pid, teamVal, esp, aimhack)
		}
	}

	if err := tx.Commit(); err != nil {
		fmt.Printf("[HandleGameEnd] DB commit failed: %v\n", err)
		return
	}

	fmt.Printf("[HandleGameEnd] Game END: %s RESULT: %s inserted as game_id=%d, logfile=%q\n", pwport, result, gameID, logfile)
}

// AddTrackedGame stores info to trackedGames
func AddTrackedGame(pw, port, logfile string, players []PlayerHackStatus) {
	trackedGamesMu.Lock()
	defer trackedGamesMu.Unlock()

	fmt.Printf("[AddTrackedGame] Called with pw=%q, port=%q, logfile=%q, players=[", pw, port, logfile)
	for i, p := range players {
		fmt.Printf("{%d: PID=%d, IngameID=%q, Team=%q, ESP=%v, Aimhack=%d}", i, p.PID, p.IngameID, p.Team, p.ESP, p.Aimhack)
		if i < len(players)-1 {
			fmt.Print(", ")
		}
	}
	fmt.Println("]")

	before := len(trackedGames)
	trackedGames = append(trackedGames, GameTracker{
		Pw:      pw,
		Port:    port,
		Logfile: logfile,
		Players: players,
		Time:    time.Now(),
	})
	after := len(trackedGames)

	fmt.Printf("[AddTrackedGame] trackedGames size: before=%d, after=%d\n", before, after)
	fmt.Printf("[AddTrackedGame] Added tracker: Pw=%q, Port=%q, Logfile=%q, PlayersCount=%d, Time=%v\n", pw, port, logfile, len(players), trackedGames[after-1].Time)
}
