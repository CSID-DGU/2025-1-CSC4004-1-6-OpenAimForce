package handlers

import (
	"GameHost/procedures"
	"io"
	"log"
	"net/http"
)

var GlobalHostManager *procedures.HostManager // Set this in main

func JoinReportHandler(w http.ResponseWriter, r *http.Request) {
	pw := r.FormValue("pw")
	content := r.FormValue("content")
	log.Printf("join-report received: pw=%s, content=%s", pw, content)
	if GlobalHostManager != nil {
		GlobalHostManager.MarkJoinConfirmed(pw)
	}
	io.WriteString(w, "OK")
}
