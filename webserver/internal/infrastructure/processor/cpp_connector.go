package processor

import (
	"fmt"
	"github.com/jrb/cuda-learning/webserver/internal/domain"
)

// CppConnector is a stub for connecting to C++ CUDA processors
type CppConnector struct{}

// NewCppConnector creates a new C++ connector instance
func NewCppConnector() *CppConnector {
	return &CppConnector{}
}

// ProcessImage is a stub that will eventually connect to C++ CUDA kernels
func (c *CppConnector) ProcessImage(img *domain.Image, filter domain.FilterType) (*domain.Image, error) {
	fmt.Println("here we are gonna connect with cpp")
	
	// For now, just return the same image (stub implementation)
	return img, nil
}

