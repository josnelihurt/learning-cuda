//go:build linux
// +build linux

package loader

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int major;
    int minor;
    int patch;
} processor_version_t;

typedef processor_version_t (*version_fn_t)(void);
typedef bool (*init_fn_t)(const uint8_t*, int, uint8_t**, int*);
typedef void (*cleanup_fn_t)();

processor_version_t call_version_fn(void* fn_ptr) {
    version_fn_t fn = (version_fn_t)fn_ptr;
    return fn();
}

bool call_init_fn(void* fn_ptr, const uint8_t* req, int req_len, uint8_t** resp, int* resp_len) {
    init_fn_t fn = (init_fn_t)fn_ptr;
    return fn(req, req_len, resp, resp_len);
}

bool call_process_fn(void* fn_ptr, const uint8_t* req, int req_len, uint8_t** resp, int* resp_len) {
    init_fn_t fn = (init_fn_t)fn_ptr;
    return fn(req, req_len, resp, resp_len);
}

void call_cleanup_fn(void* fn_ptr) {
    cleanup_fn_t fn = (cleanup_fn_t)fn_ptr;
    fn();
}

void call_free_fn(void* fn_ptr, uint8_t* buf) {
    typedef void (*free_fn_t)(uint8_t*);
    free_fn_t fn = (free_fn_t)fn_ptr;
    fn(buf);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func dlopen(path string, mode int) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.dlopen(cPath, C.int(mode))
	if handle == nil {
		errStr := C.GoString(C.dlerror())
		return 0, fmt.Errorf("dlopen failed: %s", errStr)
	}

	return uintptr(handle), nil
}

func dlsym(handle uintptr, symbol string) (uintptr, error) {
	cSymbol := C.CString(symbol)
	defer C.free(unsafe.Pointer(cSymbol))

	C.dlerror() // Clear any existing error

	sym := C.dlsym(unsafe.Pointer(handle), cSymbol)
	if sym == nil {
		errStr := C.GoString(C.dlerror())
		if errStr != "" {
			return 0, fmt.Errorf("dlsym failed: %s", errStr)
		}
	}

	return uintptr(sym), nil
}

func dlclose(handle uintptr) error {
	if C.dlclose(unsafe.Pointer(handle)) != 0 {
		errStr := C.GoString(C.dlerror())
		return fmt.Errorf("dlclose failed: %s", errStr)
	}
	return nil
}

const (
	RtldNow = C.RTLD_NOW
)

func callVersionFn(fnPtr uintptr) string {
	version := C.call_version_fn(unsafe.Pointer(fnPtr))
	return fmt.Sprintf("%d.%d.%d", version.major, version.minor, version.patch)
}

func callInitFn(fnPtr uintptr, reqBuf *uint8, reqLen int32, respBuf **uint8, respLen *int32) bool {
	return bool(C.call_init_fn(
		unsafe.Pointer(fnPtr),
		(*C.uint8_t)(reqBuf),
		C.int(reqLen),
		(**C.uint8_t)(unsafe.Pointer(respBuf)),
		(*C.int)(unsafe.Pointer(respLen)),
	))
}

func callProcessFn(fnPtr uintptr, reqBuf *uint8, reqLen int32, respBuf **uint8, respLen *int32) bool {
	return bool(C.call_process_fn(
		unsafe.Pointer(fnPtr),
		(*C.uint8_t)(reqBuf),
		C.int(reqLen),
		(**C.uint8_t)(unsafe.Pointer(respBuf)),
		(*C.int)(unsafe.Pointer(respLen)),
	))
}

func callCleanupFn(fnPtr uintptr) {
	C.call_cleanup_fn(unsafe.Pointer(fnPtr))
}

func callFreeFn(fnPtr uintptr, buf *uint8) {
	C.call_free_fn(unsafe.Pointer(fnPtr), (*C.uint8_t)(buf))
}
