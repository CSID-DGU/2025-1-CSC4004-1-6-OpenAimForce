package procedures

import (
	"GameHost/bridge"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

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

type gameState struct {
	joinConfirmed bool
	timer         *time.Timer
}

type HostManager struct {
	mu         sync.Mutex
	basePort   int
	maxPort    int
	maxRunning int
	running    map[int]*exec.Cmd
	meta       map[string]*gameState // pwport → state
}

func NewHostManager() *HostManager {
	log.Println("[NewHostManager] Initializing HostManager")
	rand.Seed(time.Now().UnixNano())
	hm := &HostManager{
		basePort:   25000,
		maxPort:    65535,
		maxRunning: 32,
		running:    make(map[int]*exec.Cmd),
		meta:       make(map[string]*gameState),
	}
	log.Printf("[NewHostManager] Initialized with basePort=%d, maxPort=%d, maxRunning=%d", hm.basePort, hm.maxPort, hm.maxRunning)
	return hm
}

func generatePassword() string {
	const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	length := rand.Intn(17) + 16
	b := make([]byte, length)
	for i := range b {
		b[i] = chars[rand.Intn(len(chars))]
	}
	pw := string(b)
	log.Printf("[generatePassword] Generated password: %s", pw)
	return pw
}

func (hm *HostManager) CreateGame(ip, path, team1, team2 string) string {
	log.Printf("[CreateGame] Called with ip=%q, path=%q, team1=%q, team2=%q", ip, path, team1, team2)
	hm.mu.Lock()
	log.Printf("[CreateGame] Lock acquired. Currently running: %d", len(hm.running))
	if len(hm.running) >= hm.maxRunning {
		hm.mu.Unlock()
		log.Println("[CreateGame] create game failed because max instance has been met")
		return "FAIL"
	}

	var port int
	for p := hm.basePort; p <= hm.maxPort; p++ {
		if _, exists := hm.running[p]; exists {
			log.Printf("[CreateGame] Skipping port %d (already in running map)", p)
			continue
		}
		if findInstance(p) != "" {
			log.Printf("[CreateGame] Skipping port %d (already in use by Docker)", p)
			continue
		}
		port = p
		break
	}
	if port == 0 {
		hm.mu.Unlock()
		log.Println("[CreateGame] create game failed: no free port available")
		return "FAIL"
	}

	password := generatePassword()
	pwport := fmt.Sprintf("%s_%d", password, port)
	cmdStr := fmt.Sprintf(`%s %d %s %s %s > gameLogs/game_%s.log 2>&1`, path, port, password, team1, team2, pwport)
	log.Printf("[CreateGame] Will run command: %s", cmdStr)
	cmd := exec.Command("bash", "-c", cmdStr)

	err := cmd.Start()
	if err != nil {
		hm.mu.Unlock()
		log.Printf("[CreateGame] cmd.Start() failed: %v", err)
		return "FAIL"
	}

	log.Printf("[CreateGame] Started process: PID=%d for port=%d", cmd.Process.Pid, port)
	hm.running[port] = cmd
	st := &gameState{joinConfirmed: false}
	hm.meta[pwport] = st

	// Timer for join report confirmation
	st.timer = time.AfterFunc(15*time.Second, func() {
		log.Printf("[CreateGame][timer] 15s timer expired for pwport=%s", pwport)
		if !st.joinConfirmed {
			hm.TerminateGameByPwPort(pwport)
			bridge.SendToMaster("KILL_DUE_TO_TIMEOUT " + pwport)
		}
	})

	hm.mu.Unlock()
	log.Printf("[CreateGame] Unlock after game creation and timer setup (port=%d)", port)

	go func(p int, c *exec.Cmd, tag string) {
		log.Printf("[CreateGame][goroutine] Waiting for process PID=%d port=%d to finish...", c.Process.Pid, p)
		_ = c.Wait()
		log.Printf("[CreateGame][goroutine] Process PID=%d for port=%d finished.", c.Process.Pid, p)
		hm.mu.Lock()
		delete(hm.running, p)
		delete(hm.meta, tag)
		log.Printf("[CreateGame][goroutine] Removed port=%d and tag=%q from running/meta maps", p, tag)
		hm.mu.Unlock()
	}(port, cmd, pwport)

	log.Printf("[CreateGame] Returning connect info: %s %d %s", ip, port, password)
	return fmt.Sprintf("%s %d %s", ip, port, password)
}

func (hm *HostManager) MarkJoinConfirmed(pw string) {
	log.Printf("[MarkJoinConfirmed] Called with pw=%q", pw)
	hm.mu.Lock()
	defer hm.mu.Unlock()

	// Try exact match first
	if st, ok := hm.meta[pw]; ok {
		st.joinConfirmed = true
		if st.timer != nil {
			log.Printf("[MarkJoinConfirmed] Stopping join timer for pwport=%q", pw)
			st.timer.Stop()
			st.timer = nil
		}
		return
	}

	// Try prefix match
	for pwport, st := range hm.meta {
		if strings.HasPrefix(pwport, pw) {
			log.Printf("[MarkJoinConfirmed] Prefix match for pwport=%q", pwport)
			st.joinConfirmed = true
			if st.timer != nil {
				log.Printf("[MarkJoinConfirmed] Stopping join timer for pwport=%q", pwport)
				st.timer.Stop()
				st.timer = nil
			}
			return
		}
	}

	log.Printf("[MarkJoinConfirmed] No state found for pw=%q", pw)
}

func findInstance(port int) string {
	log.Printf("[findInstance] Called for port=%d", port)
	out, err := exec.Command("docker", "ps", "--format", "{{.ID}} {{.Ports}}").Output()
	if err != nil {
		log.Printf("[findInstance] docker ps failed: %v", err)
		return ""
	}
	portStr := strconv.Itoa(port) + "->28763"
	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		log.Printf("[findInstance] Checking line: %q", line)
		if strings.Contains(line, portStr) {
			fields := strings.Fields(line)
			if len(fields) > 0 {
				log.Printf("[findInstance] Found container %s for port %d", fields[0], port)
				return fields[0]
			}
		}
	}
	log.Printf("[findInstance] No container found for port %d", port)
	return ""
}

func ensureStop(port int) {
	log.Printf("[ensureStop] Called for port=%d", port)
	containerID := findInstance(port)
	if containerID != "" {
		log.Printf("[ensureStop] Stopping container %s for port %d", containerID, port)
		cmd := exec.Command("docker", "stop", containerID)
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Printf("[ensureStop] docker stop failed: %v, output: %s", err, out)
		} else {
			log.Printf("[ensureStop] docker stop output: %s", out)
		}
	} else {
		log.Printf("[ensureStop] No container found to stop for port %d", port)
	}
}

func (hm *HostManager) TerminateGame(port int) {
	log.Printf("[TerminateGame] Called for port=%d", port)
	hm.mu.Lock()
	defer hm.mu.Unlock()

	cmd, exists := hm.running[port]
	log.Printf("[TerminateGame] Exists? %v, cmd: %+v", exists, cmd)
	if !exists || cmd.Process == nil {
		log.Printf("[TerminateGame] Not found in running, or no process, calling ensureStop for port %d", port)
		ensureStop(port)
		return
	}
	log.Printf("[TerminateGame] Sending os.Interrupt to process for port %d", port)
	_ = cmd.Process.Signal(os.Interrupt)
	delete(hm.running, port)
	log.Printf("[TerminateGame] Deleted port %d from running", port)
	ensureStop(port)
}

func (hm *HostManager) TerminateGameByPwPort(pwport string) bool {
	log.Printf("[TerminateGameByPwPort] Called with pwport=%q", pwport)
	hm.mu.Lock()
	defer hm.mu.Unlock()

	split := strings.Split(pwport, "_")
	log.Printf("[TerminateGameByPwPort] Split pwport: %#v", split)
	if len(split) != 2 {
		log.Printf("[TerminateGameByPwPort] Invalid pwport format")
		return false
	}
	port, err := strconv.Atoi(split[1])
	if err != nil {
		log.Printf("[TerminateGameByPwPort] Failed to parse port: %v", err)
		return false
	}
	cmd, ok := hm.running[port]
	log.Printf("[TerminateGameByPwPort] running map ok=%v, cmd=%+v", ok, cmd)
	if ok && cmd.Process != nil {
		log.Printf("[TerminateGameByPwPort] Sending os.Interrupt to process for port %d", port)
		_ = cmd.Process.Signal(os.Interrupt)
	} else {
		log.Printf("[TerminateGameByPwPort] No cmd.Process to signal for port %d", port)
	}
	delete(hm.running, port)
	delete(hm.meta, pwport)
	log.Printf("[TerminateGameByPwPort] Deleted port %d from running and %q from meta", port, pwport)
	ensureStop(port)
	return true
}

func TrackGame(pw, port string, players []PlayerHackStatus) {
	trackedGamesMu.Lock()
	defer trackedGamesMu.Unlock()
	trackedGames = append(trackedGames, GameTracker{
		Pw:      pw,
		Port:    port,
		Logfile: fmt.Sprintf("%s_%s.log", pw, port),
		Players: players,
		Time:    time.Now(),
	})
}
