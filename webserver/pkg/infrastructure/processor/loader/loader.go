package loader

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"google.golang.org/protobuf/proto"
)

const CurrentAPIVersion = "2.0.0"

type Loader struct {
	handle       uintptr
	libraryPath  string
	apiVersion   string
	capabilities *pb.LibraryCapabilities

	apiVersionFn      uintptr
	initFn            uintptr
	cleanupFn         uintptr
	processImageFn    uintptr
	getCapabilitiesFn uintptr
	freeResponseFn    uintptr
}

func NewLoader(libraryPath string) (*Loader, error) {
	handle, err := dlopen(libraryPath, RtldNow)
	if err != nil {
		return nil, fmt.Errorf("dlopen failed for %s: %w", libraryPath, err)
	}

	loader := &Loader{
		handle:      handle,
		libraryPath: libraryPath,
	}

	if resolveErr := loader.resolveSymbols(); resolveErr != nil {
		_ = dlclose(handle) //nolint:errcheck // Best effort cleanup
		return nil, resolveErr
	}

	version, err := loader.getAPIVersion()
	if err != nil {
		_ = dlclose(handle) //nolint:errcheck // Best effort cleanup
		return nil, err
	}
	loader.apiVersion = version

	if !isCompatible(CurrentAPIVersion, version) {
		_ = dlclose(handle) //nolint:errcheck // Best effort cleanup
		return nil, fmt.Errorf("API version mismatch: loader=%s, library=%s",
			CurrentAPIVersion, version)
	}

	caps, err := loader.discoverCapabilities()
	if err != nil {
		_ = dlclose(handle) //nolint:errcheck // Best effort cleanup
		return nil, err
	}
	loader.capabilities = caps

	return loader, nil
}

func (l *Loader) resolveSymbols() error {
	symbols := map[string]*uintptr{
		"processor_api_version":      &l.apiVersionFn,
		"processor_init":             &l.initFn,
		"processor_cleanup":          &l.cleanupFn,
		"processor_process_image":    &l.processImageFn,
		"processor_get_capabilities": &l.getCapabilitiesFn,
		"processor_free_response":    &l.freeResponseFn,
	}

	for name, ptr := range symbols {
		sym, err := dlsym(l.handle, name)
		if err != nil {
			return fmt.Errorf("failed to resolve symbol %s: %w", name, err)
		}
		if sym == 0 {
			return fmt.Errorf("symbol %s resolved to null pointer", name)
		}
		*ptr = sym
	}

	return nil
}

func (l *Loader) getAPIVersion() (string, error) {
	version := callVersionFn(l.apiVersionFn)
	if version == "" {
		return "", fmt.Errorf("processor_api_version returned empty")
	}
	return version, nil
}

// callCFunction is a generic helper for calling C functions with common error handling
func (l *Loader) callCFunction(
	req interface{},
	callFn func(*uint8, int32, **uint8, *int32) bool,
	responseType string,
	unmarshalFunc func([]byte) (interface{}, error),
) (interface{}, error) {
	// Marshal request
	protoMsg, ok := req.(proto.Message)
	if !ok {
		return nil, fmt.Errorf("request is not a proto.Message")
	}
	requestBytes, err := proto.Marshal(protoMsg)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal %s: %w", responseType, err)
	}

	var responsePtr *uint8
	var responseLen int32

	var reqPtr *uint8
	if len(requestBytes) > 0 {
		reqPtr = &requestBytes[0]
	}

	success := callFn(reqPtr, int32(len(requestBytes)), &responsePtr, &responseLen)

	if responsePtr != nil {
		defer l.freeResponse(responsePtr)
	}

	if responseLen <= 0 {
		return nil, fmt.Errorf("%s returned empty response", responseType)
	}

	responseBytes := unsafe.Slice(responsePtr, responseLen)

	response, err := unmarshalFunc(responseBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal %s: %w", responseType, err)
	}

	// Check for errors in the response
	if !success {
		return nil, fmt.Errorf("%s failed", responseType)
	}

	return response, nil
}

// validateAndSetAPIVersion validates and sets the API version for a request
func (l *Loader) validateAndSetAPIVersion(req interface{}) error {
	switch r := req.(type) {
	case *pb.InitRequest:
		if r.ApiVersion == "" {
			r.ApiVersion = CurrentAPIVersion
		}
		if r.ApiVersion != CurrentAPIVersion {
			return fmt.Errorf("request API version mismatch: got %s, expected %s",
				r.ApiVersion, CurrentAPIVersion)
		}
	case *pb.ProcessImageRequest:
		if r.ApiVersion == "" {
			r.ApiVersion = CurrentAPIVersion
		}
		if r.ApiVersion != CurrentAPIVersion {
			return fmt.Errorf("request API version mismatch: got %s, expected %s",
				r.ApiVersion, CurrentAPIVersion)
		}
	case *pb.GetCapabilitiesRequest:
		if r.ApiVersion == "" {
			r.ApiVersion = CurrentAPIVersion
		}
	default:
		return fmt.Errorf("unsupported request type")
	}
	return nil
}

// callCFunctionWithResponseType is a generic helper for calling C functions with response type handling
func (l *Loader) callCFunctionWithResponseType(
	req interface{},
	callFn func(*uint8, int32, **uint8, *int32) bool,
	responseType string,
	createResponse func() proto.Message,
	errorMessage string,
) (proto.Message, error) {
	if err := l.validateAndSetAPIVersion(req); err != nil {
		return nil, err
	}

	response, err := l.callCFunction(
		req,
		callFn,
		responseType,
		func(data []byte) (interface{}, error) {
			resp := createResponse()
			err := proto.Unmarshal(data, resp)
			if err != nil {
				return nil, err
			}
			// Check if response has Code field and it's not 0
			if codeResp, ok := resp.(interface{ GetCode() int32 }); ok && codeResp.GetCode() != 0 {
				if msgResp, ok := resp.(interface{ GetMessage() string }); ok {
					return nil, fmt.Errorf("%s: %s", errorMessage, msgResp.GetMessage())
				}
				return nil, fmt.Errorf("%s failed", errorMessage)
			}
			return resp, nil
		},
	)
	if err != nil {
		return nil, err
	}

	result, ok := response.(proto.Message)
	if !ok {
		return nil, fmt.Errorf("unexpected response type for %s", responseType)
	}
	return result, nil
}

func (l *Loader) Init(req *pb.InitRequest) (*pb.InitResponse, error) {
	response, err := l.callCFunctionWithResponseType(
		req,
		func(reqPtr *uint8, reqLen int32, respPtr **uint8, respLen *int32) bool {
			return callInitFn(l.initFn, reqPtr, reqLen, respPtr, respLen)
		},
		"InitResponse",
		func() proto.Message { return &pb.InitResponse{} },
		"init failed",
	)
	if err != nil {
		return nil, err
	}
	initResp, ok := response.(*pb.InitResponse)
	if !ok {
		return nil, fmt.Errorf("unexpected response type for Init")
	}
	return initResp, nil
}

func (l *Loader) ProcessImage(req *pb.ProcessImageRequest) (*pb.ProcessImageResponse, error) {
	response, err := l.callCFunctionWithResponseType(
		req,
		func(reqPtr *uint8, reqLen int32, respPtr **uint8, respLen *int32) bool {
			return callProcessFn(l.processImageFn, reqPtr, reqLen, respPtr, respLen)
		},
		"ProcessImageResponse",
		func() proto.Message { return &pb.ProcessImageResponse{} },
		"process_image failed",
	)
	if err != nil {
		return nil, err
	}
	procResp, ok := response.(*pb.ProcessImageResponse)
	if !ok {
		return nil, fmt.Errorf("unexpected response type for ProcessImage")
	}
	return procResp, nil
}

func (l *Loader) GetCapabilities(req *pb.GetCapabilitiesRequest) (*pb.GetCapabilitiesResponse, error) {
	if err := l.validateAndSetAPIVersion(req); err != nil {
		return nil, err
	}

	requestBytes, err := proto.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal GetCapabilitiesRequest: %w", err)
	}

	var responsePtr *uint8
	var responseLen int32

	var reqPtr *uint8
	if len(requestBytes) > 0 {
		reqPtr = &requestBytes[0]
	}

	success := callProcessFn(l.getCapabilitiesFn, reqPtr, int32(len(requestBytes)), &responsePtr, &responseLen)

	if responsePtr != nil {
		defer l.freeResponse(responsePtr)
	}

	if responseLen <= 0 {
		return nil, fmt.Errorf("get_capabilities returned empty response")
	}

	responseBytes := unsafe.Slice(responsePtr, responseLen)

	capsResp := &pb.GetCapabilitiesResponse{}
	if err := proto.Unmarshal(responseBytes, capsResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal GetCapabilitiesResponse: %w", err)
	}

	if !success || capsResp.Code != 0 {
		return nil, fmt.Errorf("get_capabilities failed: %s", capsResp.Message)
	}

	return capsResp, nil
}

func (l *Loader) discoverCapabilities() (*pb.LibraryCapabilities, error) {
	req := &pb.GetCapabilitiesRequest{
		ApiVersion: CurrentAPIVersion,
	}

	resp, err := l.GetCapabilities(req)
	if err != nil {
		return nil, err
	}

	return resp.Capabilities, nil
}

func (l *Loader) freeResponse(ptr *uint8) {
	callFreeFn(l.freeResponseFn, ptr)
}

func (l *Loader) Cleanup() {
	callCleanupFn(l.cleanupFn)

	if l.handle != 0 {
		_ = dlclose(l.handle) //nolint:errcheck // Best effort cleanup
		l.handle = 0
	}
}

func (l *Loader) CachedCapabilities() *pb.LibraryCapabilities {
	return l.capabilities
}

func (l *Loader) GetVersion() string {
	return l.apiVersion
}

func (l *Loader) IsCompatibleWith(apiVersion string) bool {
	return isCompatible(apiVersion, l.apiVersion)
}

func isCompatible(v1, v2 string) bool {
	return getMajorVersion(v1) == getMajorVersion(v2)
}

func getMajorVersion(version string) int {
	parts := strings.Split(version, ".")
	if len(parts) == 0 {
		return 0
	}
	major, _ := strconv.Atoi(parts[0]) //nolint:errcheck // Fallback to 0 on error is fine
	return major
}
