package processor

/*
#cgo CFLAGS: -I${SRCDIR}/../../../..
#cgo LDFLAGS: -L${SRCDIR}/../../../../bazel-bin/cpp_accelerator/ports/cgo -lcgo_api
#include "cpp_accelerator/ports/cgo/cgo_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/domain"
	"google.golang.org/protobuf/proto"
)

// CppConnector connects to C++ CUDA processors via CGO
type CppConnector struct{}

// NewCppConnector creates a new C++ connector instance
func NewCppConnector() *CppConnector {
	return &CppConnector{}
}

// init initializes the CUDA context when the package is loaded
func init() {
	// Create initialization request
	initReq := &pb.InitRequest{
		CudaDeviceId: 0, // Default CUDA device
	}

	// Marshal to bytes
	reqBytes, err := proto.Marshal(initReq)
	if err != nil {
		panic(fmt.Sprintf("Failed to marshal InitRequest: %v", err))
	}

	// Call C++ initialization
	var response *C.uint8_t
	var responseLen C.int
	
	// Safe pointer handling for CGO
	var reqPtr *C.uint8_t
	if len(reqBytes) > 0 {
		reqPtr = (*C.uint8_t)(unsafe.Pointer(&reqBytes[0]))
	}

	success := C.CudaInit(
		reqPtr,
		C.int(len(reqBytes)),
		&response,
		&responseLen,
	)

	// Always free the response
	defer C.FreeResponse(response)

	if !success {
		// Parse error response
		initResp := &pb.InitResponse{}
		respBytes := C.GoBytes(unsafe.Pointer(response), responseLen)
		if err := proto.Unmarshal(respBytes, initResp); err == nil {
			panic(fmt.Sprintf("CUDA initialization failed: %s", initResp.Message))
		}
		panic("CUDA initialization failed with unknown error")
	}

	// Parse success response
	initResp := &pb.InitResponse{}
	respBytes := C.GoBytes(unsafe.Pointer(response), responseLen)
	if err := proto.Unmarshal(respBytes, initResp); err != nil {
		panic(fmt.Sprintf("Failed to parse InitResponse: %v", err))
	}

	fmt.Printf("CUDA initialized: %s\n", initResp.Message)
}

// ProcessImage processes an image using C++ CUDA or CPU kernels
func (c *CppConnector) ProcessImage(img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType) (*domain.Image, error) {
	// Handle "none" filter or empty filters - return original image without processing
	if len(filters) == 0 || (len(filters) == 1 && filters[0] == domain.FilterNone) {
		return img, nil
	}
	
	// Map filter types to protobuf
	var protoFilters []pb.FilterType
	for _, filter := range filters {
		switch filter {
		case domain.FilterNone:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_NONE)
		case domain.FilterGrayscale:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		default:
			return nil, fmt.Errorf("unsupported filter type: %s", filter)
		}
	}
	
	// Map accelerator type to protobuf
	var protoAccelerator pb.AcceleratorType
	switch accelerator {
	case domain.AcceleratorGPU:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_GPU
	case domain.AcceleratorCPU:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	default:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_GPU // Default to GPU
	}
	
	// Map grayscale type to protobuf
	var protoGrayscaleType pb.GrayscaleType
	switch grayscaleType {
	case domain.GrayscaleBT601:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case domain.GrayscaleBT709:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case domain.GrayscaleAverage:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case domain.GrayscaleLightness:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case domain.GrayscaleLuminosity:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601 // Default to BT601
	}

	// Create process request
	procReq := &pb.ProcessImageRequest{
		ImageData:     img.Data,
		Width:         int32(img.Width),
		Height:        int32(img.Height),
		Channels:      int32(4), // Assuming RGBA format
		Filters:       protoFilters,
		Accelerator:   protoAccelerator,
		GrayscaleType: protoGrayscaleType,
	}

	// Marshal to bytes
	reqBytes, err := proto.Marshal(procReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call C++ processing
	var response *C.uint8_t
	var responseLen C.int
	
	// Safe pointer handling for CGO
	var reqPtr *C.uint8_t
	if len(reqBytes) > 0 {
		reqPtr = (*C.uint8_t)(unsafe.Pointer(&reqBytes[0]))
	}

	success := C.ProcessImage(
		reqPtr,
		C.int(len(reqBytes)),
		&response,
		&responseLen,
	)

	// Always free the response
	defer C.FreeResponse(response)

	// Parse response
	procResp := &pb.ProcessImageResponse{}
	respBytes := C.GoBytes(unsafe.Pointer(response), responseLen)
	if err := proto.Unmarshal(respBytes, procResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if !success || procResp.Code != 0 {
		return nil, fmt.Errorf("processing failed: %s", procResp.Message)
	}

	// Build result image
	result := &domain.Image{
		Data:   procResp.ImageData,
		Width:  int(procResp.Width),
		Height: int(procResp.Height),
		Format: img.Format, // Preserve original format metadata
	}

	return result, nil
}
