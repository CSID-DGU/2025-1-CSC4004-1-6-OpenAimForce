package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"matchmaker/models"
	"net/http"
	"os"
	"strings"
	"time"

	"matchmaker/handlers"

	_ "github.com/go-sql-driver/mysql"
	"github.com/gorilla/mux"
)

type DBConfig struct {
	Host     string `json:"host"`
	User     string `json:"user"`
	Password string `json:"password"`
	DBName   string `json:"dbname"`
}

func loadDBConfig() DBConfig {
	file, _ := os.ReadFile("config/db_config.json")
	var cfg DBConfig
	json.Unmarshal(file, &cfg)
	return cfg
}

func loadJWTSecret() string {
	data, err := os.ReadFile("config/jwt_secret.txt")
	if err != nil {
		log.Fatal("Missing JWT secret:", err)
	}
	return strings.TrimSpace(string(data))
}
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func loadPeriods(filename string) [][2]time.Time {
        file, err := os.Open(filename)
        if err != nil {
                log.Printf("[File Error] %v\n", err)
                return nil
        }
        defer file.Close()

        scanner := bufio.NewScanner(file)
        var periods [][2]time.Time
        layout := "2006-01-02 15:04"
        loc, err := time.LoadLocation("Asia/Seoul")
        if err != nil {
                log.Printf("[TZ Load Error] %v\n", err)
                return nil
        }
        for scanner.Scan() {
                line := scanner.Text()
                parts := strings.SplitN(line, ",", 2)
                if len(parts) != 2 {
                        log.Printf("[Split Error] line: '%s'\n", line)
                        continue
                }
                startStr := strings.TrimSpace(parts[0])
                endStr := strings.TrimSpace(parts[1])
                start, err1 := time.ParseInLocation(layout, startStr, loc)
                end, err2 := time.ParseInLocation(layout, endStr, loc)
                if err1 == nil && err2 == nil {
                        log.Printf("[Parsed OK] Allowed: %s ~ %s\n", start.Format(layout), end.Format(layout))
                        periods = append(periods, [2]time.Time{start, end})
                } else {
                        log.Printf("[Parse Error] line: '%s' (start: '%s' [%v], end: '%s' [%v])\n", line, startStr, err1, endStr, err2)
                }
        }
        return periods
}


func splitOnComma(s string) []string {
	var a, b string
	fmt.Sscanf(s, "%[^,],%s", &a, &b)
	return []string{a, b}
}

func main() {
	// Parse --team-size argument
	teamSize := flag.Int("team-size", 1, "number of players per team")
	flag.Parse()

	if *teamSize < 1 || *teamSize > 10 {
		log.Fatal("Invalid --team-size: must be between 2 and 10")
	}
	matchCount := *teamSize * 2

	cfg := loadDBConfig()
	periods := loadPeriods("config/date.txt")
	dsn := cfg.User + ":" + cfg.Password + "@tcp(" + cfg.Host + ")/" + cfg.DBName + "?parseTime=true"
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal("DB error:", err)
	}
	handlers.SetDBQueue(db)
	handlers.SetDBHost(db)
	models.SetDB(db)
	secret := loadJWTSecret()

	handlers.SetAllowedPeriods(periods)
	go handlers.StartQueueManager(matchCount)

	r := mux.NewRouter()

	// A: Register handlers
	r.HandleFunc("/signup", handlers.SignUp(db)).Methods("POST")
	r.HandleFunc("/session/login", handlers.LoginWeb(db)).Methods("POST")
	r.HandleFunc("/api/login", handlers.LoginJWT(db, secret)).Methods("POST")
	r.HandleFunc("/ladder", handlers.GetLadder(db)).Methods("GET")
	r.HandleFunc("/ws", handlers.HandleIncomingGameHost)
	r.HandleFunc("/queue/start", handlers.QueueHandler(secret)).Methods("GET")

	// Serve static files under /pages/*
	r.PathPrefix("/").Handler(http.FileServer(http.Dir("pages/")))

	// Default route: serve guide.html when accessing "/"
	r.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "pages/guide.html")
	})

	// B: Middleware and server start
	r.Use(loggingMiddleware)
	log.Println("Server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))

}
