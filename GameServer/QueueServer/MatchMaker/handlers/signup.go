package handlers

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"regexp"

	"golang.org/x/crypto/bcrypt"
)

type SignupRequest struct {
	IngameID   string `json:"ingame_id"`
	Password   string `json:"password"`
	RealName   string `json:"real_name"`
	Contact    string `json:"contact"`
	AgreeTerms bool   `json:"agree_terms"`
}

var idRegex = regexp.MustCompile(`^[a-zA-Z0-9]+$`)
var pwRegex = regexp.MustCompile(`^[a-zA-Z0-9!@#$%^&*()_\-+=\[\]{}:;'"\\|<>,.?/~` + "`" + `]{8,}$`)

func SignUp(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req SignupRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !req.AgreeTerms {
			http.Error(w, `{"error":"invalid data or terms not agreed"}`, http.StatusBadRequest)
			return
		}
		if !idRegex.MatchString(req.IngameID) {
			http.Error(w, `{"error":"invalid ingame_id"}`, http.StatusBadRequest)
			return
		}
		if !pwRegex.MatchString(req.Password) {
			http.Error(w, `{"error":"invalid password"}`, http.StatusBadRequest)
			return
		}

		hashed, _ := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
		_, err := db.Exec(`INSERT INTO Player (account_type, ingame_id, password, real_name, contact, unrank)
			VALUES ('default', ?, ?, ?, ?, TRUE)`,
			req.IngameID, string(hashed), req.RealName, req.Contact)
		if err != nil {
			http.Error(w, `{"error":"user exists or DB error"}`, http.StatusConflict)
			return
		}

		w.Write([]byte(`{"success":true}`))
	}
}
