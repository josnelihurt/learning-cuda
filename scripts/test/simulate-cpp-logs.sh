#!/bin/bash

LOG_FILE="/tmp/cppaccelerator.log"
PID_FILE="/tmp/simulate-cpp-logs.pid"

# Function to write a JSON log entry
write_json_log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    
    echo "{\"timestamp\":\"$timestamp\",\"level\":\"$level\",\"message\":\"$message\",\"source\":\"cpp\"}" >> "$LOG_FILE"
}

# Function to simulate image processing
simulate_image_processing() {
    write_json_log "info" "Initializing CUDA context (device: 0)"
    write_json_log "info" "Telemetry stub initialized (OpenTelemetry C++ not fully linked yet)"
    write_json_log "info" "Initialization successful (CUDA + CPU + Telemetry)"
    
    # Simulate processing different images
    local images=("lena.png" "barbara.png" "cameraman.png" "airplane.png" "house.png")
    for image in "${images[@]}"; do
        write_json_log "info" "Processing image: $image"
        write_json_log "info" "Image dimensions: 512x512"
        write_json_log "info" "Applying CUDA acceleration"
        write_json_log "info" "Processing completed for $image"
    done
    
    write_json_log "info" "Batch processing finished"
}

# Function to simulate error scenarios
simulate_errors() {
    write_json_log "error" "CUDA device not available, falling back to CPU"
    write_json_log "warn" "Performance may be degraded without GPU acceleration"
    write_json_log "info" "CPU processing initialized"
}

# Main simulation loop
main() {
    echo "Starting C++ logs simulation (PID: $$)"
    echo "$$" > "$PID_FILE"
    
    # Create log file if it doesn't exist
    touch "$LOG_FILE"
    
    local iteration=1
    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === C++ Logs Simulation Iteration $iteration ==="
        
        # Simulate normal processing
        simulate_image_processing
        
        # Occasionally simulate errors
        if [ $((iteration % 3)) -eq 0 ]; then
            simulate_errors
        fi
        
        # Wait before next iteration
        sleep 30
        iteration=$((iteration + 1))
    done
}

# Cleanup function
cleanup() {
    echo "Stopping C++ logs simulation..."
    rm -f "$PID_FILE"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start simulation
main
