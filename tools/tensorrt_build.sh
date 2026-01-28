#!/bin/bash
#
# TensorRT Engine Build Script
# Automates TRT engine compilation with proper error handling
#
# Usage:
#   ./tensorrt_build.sh <model.onnx> [output.engine]
#
# Features:
# - Checks for required tools
# - Validates input files
# - Timeout protection
# - Detailed logging
# - Error handling
# - Supports FP16/INT8 builds
# - Outputs build statistics
# Disclaimer: This script is for educational purposes only. 
# It demonstrates best practices for automating TensorRT engine builds.
# Author: Hamna

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
TIMEOUT=600  # 10 minutes
LOG_DIR="./trt_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/trt_build_$(date +%Y%m%d_%H%M%S).log"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 <model.onnx> [output.engine]

Arguments:
    model.onnx      Input ONNX model file
    output.engine   Output TensorRT engine file (optional, default: model.engine)

Options:
    -h, --help      Show this help message
    --fp16          Enable FP16 precision
    --int8          Enable INT8 precision
    --timeout SEC   Build timeout in seconds (default: $TIMEOUT)

Examples:
    $0 resnet50.onnx
    $0 resnet50.onnx resnet50_fp16.engine --fp16
    $0 yolov5.onnx yolov5_int8.engine --int8 --timeout 1800

EOF
    exit 1
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
fi

MODEL_FILE=""
ENGINE_FILE=""
PRECISION_FLAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --fp16)
            PRECISION_FLAGS="$PRECISION_FLAGS --fp16"
            shift
            ;;
        --int8)
            PRECISION_FLAGS="$PRECISION_FLAGS --int8"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            if [ -z "$MODEL_FILE" ]; then
                MODEL_FILE="$1"
            elif [ -z "$ENGINE_FILE" ]; then
                ENGINE_FILE="$1"
            else
                log_error "Unknown argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Set default engine name if not provided
if [ -z "$ENGINE_FILE" ]; then
    ENGINE_FILE="${MODEL_FILE%.onnx}.engine"
fi

# Validation
log_info "=========================================="
log_info "TensorRT Engine Build"
log_info "=========================================="
log_info "Timestamp: $(date)"
log_info "Model:     $MODEL_FILE"
log_info "Engine:    $ENGINE_FILE"
log_info "Timeout:   ${TIMEOUT}s"
log_info "Precision: ${PRECISION_FLAGS:-FP32 (default)}"
log_info "Log file:  $LOG_FILE"
log_info "=========================================="

# Check if trtexec exists
if [ ! -f "$TRTEXEC" ]; then
    log_error "trtexec not found at $TRTEXEC"
    log_error "Please install TensorRT or update TRTEXEC path"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    log_error "Model file not found: $MODEL_FILE"
    exit 1
fi

# Check file size
MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
log_info "Model size: $MODEL_SIZE"

# Build engine
log_info "Starting TensorRT engine build..."
log_info "This may take several minutes..."

# Build command
BUILD_CMD="timeout $TIMEOUT $TRTEXEC \
    --onnx=$MODEL_FILE \
    --saveEngine=$ENGINE_FILE \
    $PRECISION_FLAGS \
    --verbose"

log_info "Command: $BUILD_CMD"
log_info ""

# Execute with timeout
if eval "$BUILD_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    BUILD_EXIT_CODE=${PIPESTATUS[0]}
else
    BUILD_EXIT_CODE=${PIPESTATUS[0]}
fi

log_info ""
log_info "=========================================="

# Check result
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    log_info "SUCCESS: Engine built successfully"
    log_info "Output: $ENGINE_FILE"
    
    if [ -f "$ENGINE_FILE" ]; then
        ENGINE_SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
        log_info "Engine size: $ENGINE_SIZE"
        log_info ""
        ls -lh "$ENGINE_FILE" | tee -a "$LOG_FILE"
    fi
    
    log_info "=========================================="
    exit 0

elif [ $BUILD_EXIT_CODE -eq 124 ]; then
    log_error "Build TIMEOUT after ${TIMEOUT}s"
    log_warn "Consider increasing timeout or optimizing model"
    log_info "=========================================="
    exit 1

else
    log_error "Build FAILED with exit code $BUILD_EXIT_CODE"
    log_error "Check log file for details: $LOG_FILE"
    log_info "=========================================="
    exit 1
fi