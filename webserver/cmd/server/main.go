package main

import (
	"log"

	"github.com/jrb/cuda-learning/webserver/internal/app"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/infrastructure/processor"
)

func main() {
	config := app.LoadConfig()
	
	cppConnector := processor.NewCppConnector()
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)
	
	server := app.New(config, processImageUseCase)
	
	if err := server.Run(); err != nil {
		log.Fatal(err)
	}
}

