package handlers

import (
	"GameHost/bridge"
	"GameHost/procedures"
	"fmt"
	"io"
	"log"
	"net/http"
)

func EndReportHandler(hostMgr *procedures.HostManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		pw := r.FormValue("pw")
		content := r.FormValue("content")
		log.Printf("end-report received: pw=%s, content=%s", pw, content)
		bridge.SendToMaster(fmt.Sprintf("GAMEEND %s %s", pw, content))

		// Ensure termination
		hostMgr.TerminateGameByPwPort(pw + "_25000") // Update if your port is dynamic

		io.WriteString(w, "OK")
	}
}
