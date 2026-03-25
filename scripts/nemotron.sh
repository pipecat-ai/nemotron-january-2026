#!/bin/bash
# nemotron.sh - Host-side script to manage the Nemotron unified container
#
# This script runs on the host machine and manages the Docker container lifecycle.
# It provides commands for starting, stopping, viewing logs, and checking status.
#
# Usage:
#   ./scripts/nemotron.sh start [OPTIONS]     Start the container
#   ./scripts/nemotron.sh stop                Stop the container
#   ./scripts/nemotron.sh restart [OPTIONS]   Restart the container
#   ./scripts/nemotron.sh status              Show container and service status
#   ./scripts/nemotron.sh logs [SERVICE]      View logs (asr, tts, llm, or all)
#   ./scripts/nemotron.sh shell               Open a shell in the container
#   ./scripts/nemotron.sh help                Show this help message
#
# Start Options:
#   --mode MODE          LLM mode: llamacpp-q8 (default), llamacpp-q4, vllm, vllm-fp8
#   --model PATH         Path to model file (GGUF for llamacpp, HF model for vllm)
#   --no-asr             Disable ASR service
#   --no-tts             Disable TTS service
#   --no-llm             Disable LLM service
#   --detach, -d         Run in background (default)
#   --foreground, -f     Run in foreground (attach to container)
#
# Examples:
#   ./scripts/nemotron.sh start --model /path/to/Q8.gguf
#   ./scripts/nemotron.sh start --mode vllm --model nvidia/model-name
#   ./scripts/nemotron.sh start --no-llm    # ASR + TTS only
#   ./scripts/nemotron.sh logs llm          # View LLM logs
#   ./scripts/nemotron.sh logs              # View all logs interleaved

set -e

# =============================================================================
# Configuration
# =============================================================================
CONTAINER_NAME="${NEMOTRON_CONTAINER_NAME:-nemotron}"
IMAGE_NAME="${NEMOTRON_IMAGE:-nemotron-unified:cuda13}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default settings
LLAMA_MODEL=""
VLLM_MODEL=""
ENABLE_ASR="true"
ENABLE_TTS="true"
ENABLE_LLM="true"
DETACH="true"

# Default model paths (auto-detected from HuggingFace cache)
DEFAULT_Q8_MODEL="$(find "$HOME/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF" -name "*Q8*.gguf" 2>/dev/null | head -1)"
DEFAULT_Q4_MODEL="$(find "$HOME/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF" -name "*Q4*.gguf" 2>/dev/null | head -1)"
DEFAULT_VLLM_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_VLLM_FP8_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# HuggingFace model cache paths for ASR and TTS (auto-downloaded on first run)
HF_CACHE_ASR="$HOME/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b"
HF_CACHE_TTS="$HOME/.cache/huggingface/hub/models--nvidia--magpie_tts_multilingual_357m"

# Auto-detect LLM mode based on available models (prefer Q8 if available)
if [[ -n "$DEFAULT_Q8_MODEL" ]]; then
    LLM_MODE="llamacpp-q8"
elif [[ -n "$DEFAULT_Q4_MODEL" ]]; then
    LLM_MODE="llamacpp-q4"
else
    LLM_MODE="llamacpp-q8"  # Fallback, will error later if no model found
fi

# =============================================================================
# Helper functions
# =============================================================================
print_usage() {
    cat << 'EOF'
Nemotron Container Manager

Usage:
  ./scripts/nemotron.sh COMMAND [OPTIONS]

Commands:
  start [OPTIONS]     Start the container
  stop                Stop the container
  restart [OPTIONS]   Restart the container
  status              Show container and service status
  logs [SERVICE]      View logs (asr, tts, llm, or all)
  shell               Open a shell in the container
  help                Show this help message

Start Options:
  --mode MODE         LLM mode: llamacpp-q8 (default), llamacpp-q4, vllm, vllm-fp8
  --model PATH        Path to model (GGUF for llamacpp, HF id/path for vllm)
  --no-asr            Disable ASR service
  --no-tts            Disable TTS service
  --no-llm            Disable LLM service
  --detach, -d        Run in background (default)
  --foreground, -f    Run in foreground

Examples:
  # Start with default Q8 model
  ./scripts/nemotron.sh start --model ~/.cache/huggingface/.../Q8_0.gguf

  # Start with vLLM
  ./scripts/nemotron.sh start --mode vllm --model nvidia/model-name
  ./scripts/nemotron.sh start --mode vllm-fp8 --model nvidia/model-name

  # Start ASR + TTS only (no LLM)
  ./scripts/nemotron.sh start --no-llm

  # View LLM logs
  ./scripts/nemotron.sh logs llm

  # Follow all logs
  ./scripts/nemotron.sh logs

Environment Variables:
  NEMOTRON_CONTAINER_NAME   Container name (default: nemotron)
  NEMOTRON_IMAGE            Docker image (default: nemotron-unified:cuda13)
  HUGGINGFACE_ACCESS_TOKEN  HuggingFace token for gated models
EOF
}

is_container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

is_container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is not installed or not in PATH"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        echo "ERROR: Docker daemon is not running or you don't have permission"
        exit 1
    fi
}

# =============================================================================
# Command: start
# =============================================================================
cmd_start() {
    check_docker

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                LLM_MODE="$2"
                shift 2
                ;;
            --model)
                if [[ "$LLM_MODE" == vllm ]]; then
                    VLLM_MODEL="$2"
                else
                    LLAMA_MODEL="$2"
                fi
                shift 2
                ;;
            --no-asr)
                ENABLE_ASR="false"
                shift
                ;;
            --no-tts)
                ENABLE_TTS="false"
                shift
                ;;
            --no-llm)
                ENABLE_LLM="false"
                shift
                ;;
            --detach|-d)
                DETACH="true"
                shift
                ;;
            --foreground|-f)
                DETACH="false"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Handle --model after --mode is set
    if [[ -n "$2" ]] && [[ "$1" == "--model" ]]; then
        if [[ "$LLM_MODE" == vllm ]]; then
            VLLM_MODEL="$2"
        else
            LLAMA_MODEL="$2"
        fi
    fi

    # Check if container already exists
    if is_container_running; then
        echo "Container '$CONTAINER_NAME' is already running"
        echo "Use './scripts/nemotron.sh stop' first, or './scripts/nemotron.sh restart'"
        exit 1
    fi

    if is_container_exists; then
        echo "Removing stopped container '$CONTAINER_NAME'..."
        docker rm "$CONTAINER_NAME" > /dev/null
    fi

    # Validate model path for LLM (use defaults if not specified)
    if [[ "$ENABLE_LLM" == "true" ]]; then
        case "$LLM_MODE" in
            llamacpp-q8)
                if [[ -z "$LLAMA_MODEL" ]]; then
                    if [[ -n "$DEFAULT_Q8_MODEL" ]]; then
                        LLAMA_MODEL="$DEFAULT_Q8_MODEL"
                        echo "Using default Q8 model: $LLAMA_MODEL"
                    else
                        echo "ERROR: No Q8 model found in HuggingFace cache"
                        echo "Download with: huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
                        echo "Or specify: --model /path/to/model.gguf"
                        exit 1
                    fi
                fi
                # Expand ~ and make absolute
                LLAMA_MODEL="${LLAMA_MODEL/#\~/$HOME}"
                LLAMA_MODEL="$(cd "$(dirname "$LLAMA_MODEL")" && pwd)/$(basename "$LLAMA_MODEL")"
                if [[ ! -f "$LLAMA_MODEL" ]]; then
                    echo "WARNING: Model file not found: $LLAMA_MODEL"
                fi
                ;;
            llamacpp-q4)
                if [[ -z "$LLAMA_MODEL" ]]; then
                    if [[ -n "$DEFAULT_Q4_MODEL" ]]; then
                        LLAMA_MODEL="$DEFAULT_Q4_MODEL"
                        echo "Using default Q4 model: $LLAMA_MODEL"
                    else
                        echo "ERROR: No Q4 model found in HuggingFace cache"
                        echo "Download with: huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF"
                        echo "Or specify: --model /path/to/model.gguf"
                        exit 1
                    fi
                fi
                # Expand ~ and make absolute
                LLAMA_MODEL="${LLAMA_MODEL/#\~/$HOME}"
                LLAMA_MODEL="$(cd "$(dirname "$LLAMA_MODEL")" && pwd)/$(basename "$LLAMA_MODEL")"
                if [[ ! -f "$LLAMA_MODEL" ]]; then
                    echo "WARNING: Model file not found: $LLAMA_MODEL"
                fi
                ;;
            vllm)
                if [[ -z "$VLLM_MODEL" ]]; then
                    VLLM_MODEL="$DEFAULT_VLLM_MODEL"
                    echo "Using default vLLM model: $VLLM_MODEL"
                fi
                ;;
            vllm-fp8)
                if [[ -z "$VLLM_MODEL" ]]; then
                    VLLM_MODEL="$DEFAULT_VLLM_FP8_MODEL"
                    echo "Using default vLLM FP8 model: $VLLM_MODEL"
                fi
                ;;
            *)
                echo "ERROR: Unknown LLM mode: $LLM_MODE"
                echo "Valid modes: llamacpp-q8, llamacpp-q4, vllm, vllm-fp8"
                exit 1
                ;;
        esac
    fi

    # Detect if models need to be downloaded (first run)
    # If ASR or TTS models are not cached, use a longer timeout for download
    MODELS_TO_DOWNLOAD=""
    if [[ -z "${SERVICE_TIMEOUT:-}" ]]; then
        if [[ "$LLM_MODE" == vllm* ]]; then
            SERVICE_TIMEOUT=900
        else
            SERVICE_TIMEOUT=180
        fi
    fi

    if [[ "$ENABLE_ASR" == "true" ]] && [[ ! -d "$HF_CACHE_ASR" ]]; then
        MODELS_TO_DOWNLOAD="ASR"
    fi
    if [[ "$ENABLE_TTS" == "true" ]] && [[ ! -d "$HF_CACHE_TTS" ]]; then
        if [[ -n "$MODELS_TO_DOWNLOAD" ]]; then
            MODELS_TO_DOWNLOAD="$MODELS_TO_DOWNLOAD, TTS"
        else
            MODELS_TO_DOWNLOAD="TTS"
        fi
    fi

    if [[ -n "$MODELS_TO_DOWNLOAD" ]]; then
        SERVICE_TIMEOUT=600  # 10 minutes for model downloads
        echo "============================================"
        echo "FIRST RUN: Models will be downloaded"
        echo "============================================"
        echo "  Models to download: $MODELS_TO_DOWNLOAD"
        echo "  This may take several minutes..."
        echo "  (Subsequent runs will use cached models)"
        echo ""
        echo "  Timeout increased to ${SERVICE_TIMEOUT}s for downloads"
        echo "============================================"
        echo ""
    fi

    echo "============================================"
    echo "Starting Nemotron Container"
    echo "============================================"
    echo "  Container: $CONTAINER_NAME"
    echo "  Image: $IMAGE_NAME"
    echo "  Mode: $([ "$DETACH" == "true" ] && echo "detached" || echo "foreground")"
    echo ""
    echo "  Services:"
    echo "    ASR: $([ "$ENABLE_ASR" == "true" ] && echo "ENABLED" || echo "DISABLED")"
    echo "    TTS: $([ "$ENABLE_TTS" == "true" ] && echo "ENABLED" || echo "DISABLED")"
    echo "    LLM: $([ "$ENABLE_LLM" == "true" ] && echo "ENABLED ($LLM_MODE)" || echo "DISABLED")"
    echo "============================================"

    # Build docker run command
    # Use the explicit NVIDIA runtime because some Docker setups expose the GPU
    # reliably only when --runtime=nvidia is specified alongside --gpus all.
    # Use host network for vLLM mode to avoid DNS issues with HuggingFace.
    if [[ "$LLM_MODE" == vllm* ]]; then
        DOCKER_ARGS=(
            run
            --name "$CONTAINER_NAME"
            --runtime=nvidia
            --gpus all
            --network=host
            --ipc=host
            -v "$PROJECT_DIR:/workspace"
            -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
            -e "ENABLE_ASR=$ENABLE_ASR"
            -e "ENABLE_TTS=$ENABLE_TTS"
            -e "ENABLE_LLM=$ENABLE_LLM"
            -e "LLM_MODE=$LLM_MODE"
            -e "HF_HOME=/root/.cache/huggingface"
            -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
            -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
        )
    else
        DOCKER_ARGS=(
            run
            --name "$CONTAINER_NAME"
            --runtime=nvidia
            --gpus all
            --ipc=host
            -v "$PROJECT_DIR:/workspace"
            -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
            -p 8000:8000
            -p 8001:8001
            -p 8080:8080
            -e "ENABLE_ASR=$ENABLE_ASR"
            -e "ENABLE_TTS=$ENABLE_TTS"
            -e "ENABLE_LLM=$ENABLE_LLM"
            -e "LLM_MODE=$LLM_MODE"
            -e "HF_HOME=/root/.cache/huggingface"
            -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
            -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
        )
    fi

    # Add HuggingFace token if set
    if [[ -n "$HUGGINGFACE_ACCESS_TOKEN" ]]; then
        DOCKER_ARGS+=(-e "HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN")
        DOCKER_ARGS+=(-e "HF_TOKEN=$HUGGINGFACE_ACCESS_TOKEN")
    fi

    # vLLM-specific overrides (optional)
    if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
        DOCKER_ARGS+=(-e "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND")
    fi
    if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
        DOCKER_ARGS+=(-e "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN")
    fi
    if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
        DOCKER_ARGS+=(-e "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION")
    fi

    # Service timeout (longer for first-run model downloads)
    DOCKER_ARGS+=(-e "SERVICE_TIMEOUT=$SERVICE_TIMEOUT")

    # PyTorch memory allocator config (avoids fragmentation on 32GB GPUs)
    DOCKER_ARGS+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Add model path based on mode
    if [[ "$ENABLE_LLM" == "true" ]]; then
        case "$LLM_MODE" in
            llamacpp-q8|llamacpp-q4)
                # Convert host path to container path
                # ~/.cache/huggingface -> /root/.cache/huggingface
                CONTAINER_MODEL_PATH="${LLAMA_MODEL/$HOME\//.cache\/huggingface -> /root\/.cache\/huggingface}"
                if [[ "$LLAMA_MODEL" == "$HOME/.cache/huggingface"* ]]; then
                    CONTAINER_MODEL_PATH="/root/.cache/huggingface${LLAMA_MODEL#$HOME/.cache/huggingface}"
                else
                    # Model is outside HF cache, mount it directly
                    MODEL_DIR="$(dirname "$LLAMA_MODEL")"
                    MODEL_NAME="$(basename "$LLAMA_MODEL")"
                    DOCKER_ARGS+=(-v "$MODEL_DIR:/models:ro")
                    CONTAINER_MODEL_PATH="/models/$MODEL_NAME"
                fi
                DOCKER_ARGS+=(-e "LLAMA_MODEL=$CONTAINER_MODEL_PATH")
                ;;
            vllm|vllm-fp8)
                DOCKER_ARGS+=(-e "VLLM_MODEL=$VLLM_MODEL")
                ;;
        esac
    fi

    # Add detach flag
    if [[ "$DETACH" == "true" ]]; then
        DOCKER_ARGS+=(-d)
    else
        DOCKER_ARGS+=(-it --rm)
    fi

    # Add image and command
    DOCKER_ARGS+=(
        "$IMAGE_NAME"
        bash /workspace/scripts/start_unified.sh
    )

    # Run docker
    echo ""
    echo "Starting container..."
    docker "${DOCKER_ARGS[@]}"

    if [[ "$DETACH" == "true" ]]; then
        echo ""
        echo "Container started in background."
        echo ""
        echo "Useful commands:"
        echo "  ./scripts/nemotron.sh status    - Check service status"
        echo "  ./scripts/nemotron.sh logs      - View all logs"
        echo "  ./scripts/nemotron.sh logs llm  - View LLM logs only"
        echo "  ./scripts/nemotron.sh stop      - Stop the container"
    fi
}

# =============================================================================
# Command: stop
# =============================================================================
cmd_stop() {
    check_docker

    if ! is_container_running; then
        if is_container_exists; then
            echo "Container '$CONTAINER_NAME' exists but is not running"
            echo "Removing stopped container..."
            docker rm "$CONTAINER_NAME" > /dev/null
        else
            echo "Container '$CONTAINER_NAME' is not running"
        fi
        return 0
    fi

    echo "Stopping container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME" > /dev/null

    echo "Removing container..."
    docker rm "$CONTAINER_NAME" > /dev/null

    echo "Container stopped and removed."
}

# =============================================================================
# Command: restart
# =============================================================================
cmd_restart() {
    cmd_stop
    echo ""
    cmd_start "$@"
}

# =============================================================================
# Command: status
# =============================================================================
cmd_status() {
    check_docker

    echo "============================================"
    echo "Nemotron Container Status"
    echo "============================================"

    if ! is_container_exists; then
        echo "  Container: NOT FOUND"
        echo ""
        echo "Use './scripts/nemotron.sh start' to create the container"
        return 0
    fi

    if is_container_running; then
        echo "  Container: RUNNING"
        echo ""

        # Get container info
        CONTAINER_INFO=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
        echo "  Started: $CONTAINER_INFO"
        echo ""

        # Check service health
        echo "  Services:"

        # ASR health
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            echo "    ASR (port 8080): UP"
        else
            echo "    ASR (port 8080): DOWN or DISABLED"
        fi

        # TTS health
        if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
            echo "    TTS (port 8001): UP"
        else
            echo "    TTS (port 8001): DOWN or DISABLED"
        fi

        # LLM health
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "    LLM (port 8000): UP"
        else
            echo "    LLM (port 8000): DOWN or DISABLED"
        fi
    else
        echo "  Container: STOPPED"
        echo ""
        echo "Use './scripts/nemotron.sh start' to start the container"
    fi

    echo "============================================"
}

# =============================================================================
# Command: logs
# =============================================================================
cmd_logs() {
    check_docker

    if ! is_container_running; then
        echo "ERROR: Container '$CONTAINER_NAME' is not running"
        exit 1
    fi

    SERVICE="${1:-all}"

    case "$SERVICE" in
        asr)
            echo "=== ASR Logs (Ctrl+C to exit) ==="
            docker exec "$CONTAINER_NAME" tail -f /var/log/nemotron/asr.log
            ;;
        tts)
            echo "=== TTS Logs (Ctrl+C to exit) ==="
            docker exec "$CONTAINER_NAME" tail -f /var/log/nemotron/tts.log
            ;;
        llm)
            echo "=== LLM Logs (Ctrl+C to exit) ==="
            docker exec "$CONTAINER_NAME" tail -f /var/log/nemotron/llm.log
            ;;
        all)
            echo "=== All Logs (Ctrl+C to exit) ==="
            echo "  [ASR] = ASR service, [TTS] = TTS service, [LLM] = LLM service"
            echo ""
            # Use tail with headers, interleaved
            docker exec "$CONTAINER_NAME" bash -c '
                tail -f /var/log/nemotron/asr.log 2>/dev/null | sed "s/^/[ASR] /" &
                tail -f /var/log/nemotron/tts.log 2>/dev/null | sed "s/^/[TTS] /" &
                tail -f /var/log/nemotron/llm.log 2>/dev/null | sed "s/^/[LLM] /" &
                wait
            '
            ;;
        *)
            echo "ERROR: Unknown service: $SERVICE"
            echo "Valid services: asr, tts, llm, all"
            exit 1
            ;;
    esac
}

# =============================================================================
# Command: shell
# =============================================================================
cmd_shell() {
    check_docker

    if ! is_container_running; then
        echo "ERROR: Container '$CONTAINER_NAME' is not running"
        exit 1
    fi

    echo "Opening shell in container '$CONTAINER_NAME'..."
    docker exec -it "$CONTAINER_NAME" bash
}

# =============================================================================
# Main
# =============================================================================
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    start)
        cmd_start "$@"
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart "$@"
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$@"
        ;;
    shell)
        cmd_shell
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        print_usage
        exit 1
        ;;
esac
