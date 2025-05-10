package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"

	_ "github.com/go-sql-driver/mysql"
	"github.com/gorilla/mux"
	"matchmaker/handlers"
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

func main() {
	cfg := loadDBConfig()
	dsn := cfg.User + ":" + cfg.Password + "@tcp(" + cfg.Host + ")/" + cfg.DBName + "?parseTime=true"
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal("DB error:", err)
	}

	secret := loadJWTSecret()
	r := mux.NewRouter()

	r.HandleFunc("/signup", handlers.SignUp(db)).Methods("POST")
	r.HandleFunc("/session/login", handlers.LoginWeb(db)).Methods("POST")
	r.HandleFunc("/api/login", handlers.LoginJWT(db, secret)).Methods("POST")

	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
}
