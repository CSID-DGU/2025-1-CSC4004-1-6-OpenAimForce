package handlers

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type LoginRequest struct {
	IngameID string `json:"ingame_id"`
	Password string `json:"password"`
}

func checkCredentials(db *sql.DB, id, pw string) (int, error) {
	var pid int
	var hashed string
	err := db.QueryRow("SELECT pid, password FROM Player WHERE ingame_id = ?", id).Scan(&pid, &hashed)
	if err != nil {
		return 0, err
	}
	if bcrypt.CompareHashAndPassword([]byte(hashed), []byte(pw)) != nil {
		return 0, sql.ErrNoRows
	}
	return pid, nil
}

// Web login: plain response (for form POSTs)
func LoginWeb(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req LoginRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Bad request", http.StatusBadRequest)
			return
		}
		pid, err := checkCredentials(db, req.IngameID, req.Password)
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}
		http.SetCookie(w, &http.Cookie{
			Name:     "pid",
			Value:    string(rune(pid)),
			Path:     "/",
			Expires:  time.Now().Add(1 * time.Hour),
			HttpOnly: true,
		})
		w.Write([]byte("Login success"))
	}
}

// JWT login: returns JSON token
func LoginJWT(db *sql.DB, secret string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req LoginRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid input"}`, http.StatusBadRequest)
			return
		}
		pid, err := checkCredentials(db, req.IngameID, req.Password)
		if err != nil {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}

		token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
			"pid": pid,
			"exp": time.Now().Add(2 * time.Hour).Unix(),
		})
		tokenString, err := token.SignedString([]byte(secret))
		if err != nil {
			http.Error(w, `{"error":"token error"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"token": tokenString,
		})
	}
}
