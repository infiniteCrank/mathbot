package db

import (
	"database/sql"
	"log"

	_ "github.com/lib/pq"
)

const DbConnStr = "postgres://mathbot:Jd912416!!@localhost:5432/my_neuralnet_db?sslmode=disable"

func ConnectDB() *sql.DB {
	db, err := sql.Open("postgres", DbConnStr)
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	return db
}
