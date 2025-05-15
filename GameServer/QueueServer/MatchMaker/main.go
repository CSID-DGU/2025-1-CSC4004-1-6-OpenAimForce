package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"strings"

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
func main() {
	// Parse --team-size argument
	teamSize := flag.Int("team-size", 1, "number of players per team")
	flag.Parse()

	if *teamSize < 1 || *teamSize > 10 {
		log.Fatal("Invalid --team-size: must be between 2 and 10")
	}
	matchCount := *teamSize * 2

	cfg := loadDBConfig()
	dsn := cfg.User + ":" + cfg.Password + "@tcp(" + cfg.Host + ")/" + cfg.DBName + "?parseTime=true"
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal("DB error:", err)
	}
	handlers.SetDB(db)

	secret := loadJWTSecret()

	go handlers.StartQueueManager(matchCount)

	r := mux.NewRouter()

	// A: Register handlers
	r.HandleFunc("/signup", handlers.SignUp(db)).Methods("POST")
	r.HandleFunc("/session/login", handlers.LoginWeb(db)).Methods("POST")
	r.HandleFunc("/api/login", handlers.LoginJWT(db, secret)).Methods("POST")
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
