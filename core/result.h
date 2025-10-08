#pragma once

#include <string>
#include <optional>

namespace jrb::core {

template<typename T = void>
struct Result {
    bool success;
    std::string message;
    int exit_code;
    std::optional<T> value;
    
    static Result ok(T val, std::string msg = "", int code = 0) {
        return Result{true, msg, code, std::move(val)};
    }
    
    static Result error(std::string msg, int code = 1) {
        return Result{false, msg, code, std::nullopt};
    }
    
    explicit operator bool() const { return success; }
};

// Especialización para void
template<>
struct Result<void> {
    bool success;
    std::string message;
    int exit_code;
    
    static Result ok(std::string msg = "", int code = 0) {
        return Result{true, msg, code};
    }
    
    static Result error(std::string msg, int code = 1) {
        return Result{false, msg, code};
    }
    
    explicit operator bool() const { return success; }
};

}  // namespace jrb::core

