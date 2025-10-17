package loader

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"google.golang.org/protobuf/proto"
)

const CurrentAPIVersion = "1.0.0"

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
	handle, err := dlopen(libraryPath, RTLD_NOW)
	if err != nil {
		return nil, fmt.Errorf("dlopen failed for %s: %w", libraryPath, err)
	}

	loader := &Loader{
		handle:      handle,
		libraryPath: libraryPath,
	}

	if err := loader.resolveSymbols(); err != nil {
		dlclose(handle)
		return nil, err
	}

	version, err := loader.getAPIVersion()
	if err != nil {
		dlclose(handle)
		return nil, err
	}
	loader.apiVersion = version

	if !isCompatible(CurrentAPIVersion, version) {
		dlclose(handle)
		return nil, fmt.Errorf("API version mismatch: loader=%s, library=%s",
			CurrentAPIVersion, version)
	}

	caps, err := loader.discoverCapabilities()
	if err != nil {
		dlclose(handle)
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

func (l *Loader) Init(req *pb.InitRequest) (*pb.InitResponse, error) {
	if req.ApiVersion == "" {
		req.ApiVersion = CurrentAPIVersion
	}

	if req.ApiVersion != CurrentAPIVersion {
		return nil, fmt.Errorf("request API version mismatch: got %s, expected %s",
			req.ApiVersion, CurrentAPIVersion)
	}

	requestBytes, err := proto.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal InitRequest: %w", err)
	}

	var responsePtr *uint8
	var responseLen int32

	var reqPtr *uint8
	if len(requestBytes) > 0 {
		reqPtr = &requestBytes[0]
	}

	success := callInitFn(l.initFn, reqPtr, int32(len(requestBytes)), &responsePtr, &responseLen)

	if responsePtr != nil {
		defer l.freeResponse(responsePtr)
	}

	if responseLen <= 0 {
		return nil, fmt.Errorf("init returned empty response")
	}

	responseBytes := unsafe.Slice(responsePtr, responseLen)

	initResp := &pb.InitResponse{}
	if err := proto.Unmarshal(responseBytes, initResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal InitResponse: %w", err)
	}

	if !success || initResp.Code != 0 {
		return nil, fmt.Errorf("init failed: %s", initResp.Message)
	}

	return initResp, nil
}

func (l *Loader) ProcessImage(req *pb.ProcessImageRequest) (*pb.ProcessImageResponse, error) {
	if req.ApiVersion == "" {
		req.ApiVersion = CurrentAPIVersion
	}

	if req.ApiVersion != CurrentAPIVersion {
		return nil, fmt.Errorf("request API version mismatch: got %s, expected %s",
			req.ApiVersion, CurrentAPIVersion)
	}

	requestBytes, err := proto.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ProcessImageRequest: %w", err)
	}

	var responsePtr *uint8
	var responseLen int32

	var reqPtr *uint8
	if len(requestBytes) > 0 {
		reqPtr = &requestBytes[0]
	}

	success := callProcessFn(l.processImageFn, reqPtr, int32(len(requestBytes)), &responsePtr, &responseLen)

	if responsePtr != nil {
		defer l.freeResponse(responsePtr)
	}

	if responseLen <= 0 {
		return nil, fmt.Errorf("process_image returned empty response")
	}

	responseBytes := unsafe.Slice(responsePtr, responseLen)

	procResp := &pb.ProcessImageResponse{}
	if err := proto.Unmarshal(responseBytes, procResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ProcessImageResponse: %w", err)
	}

	if !success || procResp.Code != 0 {
		return nil, fmt.Errorf("process_image failed: %s", procResp.Message)
	}

	return procResp, nil
}

func (l *Loader) GetCapabilities(req *pb.GetCapabilitiesRequest) (*pb.GetCapabilitiesResponse, error) {
	if req.ApiVersion == "" {
		req.ApiVersion = CurrentAPIVersion
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
		dlclose(l.handle)
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
	major, _ := strconv.Atoi(parts[0])
	return major
}
