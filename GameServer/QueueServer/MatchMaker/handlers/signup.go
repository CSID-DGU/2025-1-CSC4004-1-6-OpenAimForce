package handlers

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"

	"golang.org/x/crypto/bcrypt"
)

type SignupRequest struct {
	AccountType   string `json:"account_type"`
	IngameID      string `json:"ingame_id"`
	StudentID     int    `json:"student_id"`
	Password      string `json:"password"`
	RealName      string `json:"real_name"`
	Contact       string `json:"contact"`
	AgreeTerms    bool   `json:"agree_terms"`
	OverwatchTier int    `json:"overwatch_tier"`
	ValorantTier  int    `json:"valorant_tier"`
	EtcTier       string `json:"etc_tier"`
}

var (
	allowedAccountTypes = map[string]bool{
		"default": true, "admin": true, "club": true, "collaborator": true,
	}
	idRegex   = regexp.MustCompile(`^[a-zA-Z0-9]+$`)
	pwRegex   = regexp.MustCompile(`^[a-zA-Z0-9!@#$%^&*()_\-+=\[\]{}:;'"\\|<>,.?/~` + "`" + `]{8,}$`)
	maxLength = func(s string, n int) bool { return len(s) <= n }
)

func SignUp(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req SignupRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !req.AgreeTerms {
			http.Error(w, `{"error":"약관에 동의해주세요"}`, http.StatusBadRequest)
			return
		}

		// Account type
		if !allowedAccountTypes[req.AccountType] {
			http.Error(w, `{"error":"올바르지 않은 계정 종류입니다"}`, http.StatusBadRequest)
			return
		}

		// Ingame ID: alphanumeric + length ≤ 64
		if !idRegex.MatchString(req.IngameID) || len(req.IngameID) > 64 {
			http.Error(w, `{"error":"올바르지 않은 인게임 id 형식입니다"}`, http.StatusBadRequest)
			return
		}

		// Password format
		if !pwRegex.MatchString(req.Password) {
			http.Error(w, `{"error":"올바르지 않은 비밀번호 형식입니다"}`, http.StatusBadRequest)
			return
		}

		// Student ID: 0~25
		if req.StudentID < 0 || req.StudentID > 25 {
			http.Error(w, `{"error":"올바르지 않은 학번 형식입니다"}`, http.StatusBadRequest)
			return
		}

		// Real name: ≤ 64
		if !maxLength(req.RealName, 64) {
			http.Error(w, `{"error":"실명이 최대 길이를 초과하였습니다"}`, http.StatusBadRequest)
			return
		}

		// Contact: ≤ 128
		if !maxLength(req.Contact, 128) {
			http.Error(w, `{"error":"연락처가 최대 길이를 초과하였습니다"}`, http.StatusBadRequest)
			return
		}

		// Overwatch Tier: 1~36
		if req.OverwatchTier < 1 || req.OverwatchTier > 36 {
			http.Error(w, `{"error":"오버워치 티어의 형식이 올바르지 않습니다"}`, http.StatusBadRequest)
			return
		}

		// Valorant Tier: 1~24
		if req.ValorantTier < 1 || req.ValorantTier > 24 {
			http.Error(w, `{"error":"발로란트 티어의 형식이 올바르지 않습니다"}`, http.StatusBadRequest)
			return
		}

		// Etc tier: ≤ 255
		if !maxLength(req.EtcTier, 255) {
			http.Error(w, `{"error":"기타 티어의 길이가 최대 길이를 초과하였습니다"}`, http.StatusBadRequest)
			return
		}

		unrank := req.OverwatchTier == 1 && req.ValorantTier == 1

		hashed, _ := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)

		_, err := db.Exec(`INSERT INTO Player (
			account_type, ingame_id, student_id, password,
			real_name, contact, unrank, overwatch_tier, valorant_tier, etc_tier
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			req.AccountType, req.IngameID, req.StudentID, string(hashed),
			req.RealName, req.Contact, unrank, req.OverwatchTier, req.ValorantTier, req.EtcTier)

		if err != nil {
			if strings.Contains(err.Error(), "Duplicate entry") {
				http.Error(w, `{"error":"해당 인게임 id가 이미 사용 중입니다"}`, http.StatusConflict)
			} else {
				http.Error(w, `{"error":"알 수 없는 오류"}`, http.StatusInternalServerError)
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"success":true}`))
	}
}
