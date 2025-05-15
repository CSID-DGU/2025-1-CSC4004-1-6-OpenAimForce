package main

import (
	"GameHost/handlers"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/hello", handlers.Hello)
	http.Handle("/", http.FileServer(http.Dir("static")))

	log.Println("Server started on :24999")
	log.Fatal(http.ListenAndServe(":24999", nil))
}
