package handlers

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"net/url"
	"strconv"
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

func LoginWeb(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "폼 데이터를 읽을 수 없습니다", http.StatusBadRequest)
			return
		}

		ingameID := r.FormValue("ingame_id")
		password := r.FormValue("password")
		log.Println("Parsed form data:", ingameID, password)

		pid, err := checkCredentials(db, ingameID, password)
		if err != nil {
			http.Error(w, "아이디 또는 비밀번호가 올바르지 않습니다", http.StatusUnauthorized)
			return
		}

		var realName string
		err = db.QueryRow("SELECT real_name FROM Player WHERE pid = ?", pid).Scan(&realName)
		if err != nil {
			http.Error(w, "유저 정보를 가져오는 데 실패했습니다", http.StatusInternalServerError)
			return
		}

		http.SetCookie(w, &http.Cookie{
			Name:     "pid",
			Value:    strconv.Itoa(pid),
			Path:     "/",
			Expires:  time.Now().Add(1 * time.Hour),
			HttpOnly: true,
		})

		http.SetCookie(w, &http.Cookie{
			Name:     "real_name",
			Value:    url.QueryEscape(realName), // URL encode for safety
			Path:     "/",
			Expires:  time.Now().Add(1 * time.Hour),
			HttpOnly: false,
		})

		http.Redirect(w, r, "/guide.html", http.StatusSeeOther)
	}
}

// JWT login: returns JSON token
func LoginJWT(db *sql.DB, secret string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req LoginRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			// AssaultCube Client에서의 혹시 모를 string 오류를 방지하기 위해 JWT는 한글화하지 않음
			http.Error(w, `{"error":"invalid input"}`, http.StatusBadRequest)
			return
		}
		pid, err := checkCredentials(db, req.IngameID, req.Password)
		if err != nil {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}

		var mmr int
		err = db.QueryRow("CALL get_mmr(?)", pid).Scan(&mmr)
		if err != nil {
			http.Error(w, `{"error":"mmr fetch failed"}`, http.StatusInternalServerError)
			return
		}

		token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
			"pid": pid,
			"mmr": mmr,
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
