package handlers

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

type LadderEntry struct {
	IngameID    string  `json:"ingame_id"`
	Games       int     `json:"games"`
	TotalKills  int     `json:"kills"`
	TotalDeaths int     `json:"deaths"`
	Score       float64 `json:"score"`
	Percentage  string  `json:"percentage"`
}

type LadderResponse struct {
	Club    []LadderEntry `json:"club"`
	Default []LadderEntry `json:"default"`
}

func GetLadder(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		periodStr := r.URL.Query().Get("period")
		periodIdx, _ := strconv.Atoi(periodStr)
		logPrefix := "[Ladder] "

		//fmt.Printf("%sPeriod requested: idx=%d str='%s'\n", logPrefix, periodIdx, periodStr)

		if periodIdx < 0 || periodIdx >= len(allowedPeriods) {
			fmt.Printf("%sOut of bounds period index: %d\n", logPrefix, periodIdx)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"club":[],"default":[]}`))
			return
		}
		start, end := allowedPeriods[periodIdx][0], allowedPeriods[periodIdx][1]
		//fmt.Printf("%sRange: %v ~ %v\n", logPrefix, start, end)

		type stat struct {
			IngameID string
			Games    int
			Kills    int
			Deaths   int
			Score    float64
		}
		clubStats := make(map[string]*stat)
		defaultStats := make(map[string]*stat)

		query := `
SELECT p.ingame_id, p.account_type, gp.kills, gp.deaths, gp.esp, gp.aimhack
FROM Player p
JOIN GameParticipation gp ON gp.pid = p.pid
JOIN Game g ON g.game_id = gp.game_id
WHERE g.game_time >= ? AND g.game_time <= ?
  AND gp.quitTime IS NULL
`
		rows, err := db.Query(query, start, end)
		if err != nil {
			//fmt.Printf("%sQuery error: %v\n", logPrefix, err)
			http.Error(w, "query error", 500)
			return
		}
		defer rows.Close()

		rowCount := 0
		for rows.Next() {
			var id, typ string
			var kills, deaths int
			var esp bool
			var aimhack int
			if err := rows.Scan(&id, &typ, &kills, &deaths, &esp, &aimhack); err != nil {
				//fmt.Printf("%sRow scan error: %v\n", logPrefix, err)
				continue
			}
			rowCount++
			var m map[string]*stat
			if typ == "club" {
				m = clubStats
			} else if typ == "default" {
				m = defaultStats
			} else {
				//fmt.Printf("%sUnknown account type: %s (id=%s)\n", logPrefix, typ, id)
				continue
			}
			s, ok := m[id]
			if !ok {
				s = &stat{IngameID: id}
				m[id] = s
			}
			s.Games++
			s.Kills += kills
			s.Deaths += deaths
			mult := 1.0
			if esp || aimhack >= 2 {
				mult = 2.0
			}
			s.Score += float64(kills) / (float64(max(1, deaths)) * mult)
		}
		//fmt.Printf("%sTotal rows processed: %d\n", logPrefix, rowCount)
		//fmt.Printf("%sUnique club players: %d, default players: %d\n", logPrefix, len(clubStats), len(defaultStats))

		prepare := func(m map[string]*stat, group string) []LadderEntry {
			var out []LadderEntry
			var totalScore float64
			for _, s := range m {
				totalScore += s.Score
			}
			for _, s := range m {
				perc := 0.0
				if totalScore > 0 {
					perc = s.Score / totalScore * 100
				}
				//fmt.Printf("%s[%s] %s: Games=%d Kills=%d Deaths=%d Score=%.2f Perc=%.2f%%\n",
				//	logPrefix, group, s.IngameID, s.Games, s.Kills, s.Deaths, s.Score, perc)
				out = append(out, LadderEntry{
					IngameID:    s.IngameID,
					Games:       s.Games,
					TotalKills:  s.Kills,
					TotalDeaths: s.Deaths,
					Score:       s.Score,
					Percentage:  fmt.Sprintf("%.2f", perc),
				})
			}
			return out
		}

		resp := LadderResponse{
			Club:    prepare(clubStats, "club"),
			Default: prepare(defaultStats, "default"),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
