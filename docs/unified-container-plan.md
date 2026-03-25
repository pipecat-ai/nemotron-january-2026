# Unified Container Plan

This document describes the plan to combine the three services (ASR, TTS, LLM) currently running in two separate containers into a single unified container for NVIDIA DGX Spark (Blackwell GB10, sm_121).

## Implementation Progress

| Step | Status | Notes |
|------|--------|-------|
| 1. Create planning doc | DONE | This document |
| 2. Create Dockerfile.unified | DONE | SHA1-pinned commits |
| 3. Create scripts/start_unified.sh | DONE | Health checks + warmup |
| 4. Create scripts/nemotron.sh | DONE | Host-side wrapper |
| 5. Add GPU warmup to ASR server | DONE | Claims memory before LLM |
| 6. Build container | DONE | ~2-3 hours, 29.4GB image |
| 7. Test ASR service | DONE | WebSocket working |
| 8. Test TTS service | DONE | WebSocket streaming working |
| 9. Test LLM service (llama.cpp) | DONE | HTTP completion working |
| 10. Test LLM service (vLLM) | DONE | All 3 services running together |
| 11. Test full pipeline | DONE | LLM→TTS integration verified |
| 12. Migration from two containers | PENDING | |

### Important Notes

**HUGGINGFACE_ACCESS_TOKEN Required**: The TTS model (nvidia/magpie_tts_multilingual_357m) is a gated model requiring authentication. Set the token before starting:
```bash
export HUGGINGFACE_ACCESS_TOKEN=$(cat ~/.cache/huggingface/token)
./scripts/nemotron.sh start
```

Or ensure the token is available in the environment when running the container.

## Current Architecture (Two Containers)

```
┌─────────────────────────────────────────────────────────────────┐
│  nemotron-asr container (Dockerfile.asr-cuda13-build)           │
│  Base: nvidia/cuda:13.1.0-devel-ubuntu24.04                     │
│  ├─ PyTorch (from source, sm_121)                               │
│  ├─ torchaudio (from source)                                    │
│  ├─ NeMo (main branch, pinned commit)                           │
│  ├─ ASR server (port 8080) - Parakeet 600M                      │
│  └─ TTS server (port 8001) - Magpie 357M                        │
│  VRAM: ~5GB combined                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  llama-q4 container (vllm-llamacpp:cuda13)                      │
│  Base: vllm:cuda13-full (from Dockerfile.vllm-cuda13-build)     │
│  ├─ PyTorch (from source, sm_121)                               │
│  ├─ vLLM (from source)                                          │
│  ├─ llama.cpp (from source, sm_121a)                            │
│  └─ LLM server (port 8000) - Nemotron-3-Nano Q8                 │
│  VRAM: ~32GB (Q8 GGUF) or ~72GB (BF16 vLLM)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Problems with Two Containers
1. **Separate CUDA contexts**: Each container has its own CUDA context, preventing optimal GPU memory sharing
2. **Deployment complexity**: Two containers to manage, deploy, and monitor
3. **Resource overhead**: Duplicate PyTorch installations, separate process isolation overhead
4. **Network latency**: Services communicate over Docker network instead of localhost

## Target Architecture (Single Container)

```
┌─────────────────────────────────────────────────────────────────┐
│  nemotron-unified container (Dockerfile.unified)                │
│  Base: nvidia/cuda:13.1.0-devel-ubuntu24.04                     │
│  ├─ System dependencies (union of all requirements)             │
│  ├─ cuDNN + NCCL (from NGC PyTorch container)                   │
│  ├─ PyTorch (from source, sm_121) - SINGLE BUILD                │
│  ├─ torchaudio (from source)                                    │
│  ├─ NeMo (main branch, ASR + TTS)                               │
│  ├─ vLLM (from source) - for BF16 full-weights mode             │
│  ├─ llama.cpp (from source, sm_121a) - for GGUF quantized mode  │
│  │                                                              │
│  │  Runtime Services:                                           │
│  ├─ ASR server (port 8080) - Parakeet 600M                      │
│  ├─ TTS server (port 8001) - Magpie 357M                        │
│  └─ LLM server (port 8000) - llama.cpp OR vLLM                  │
│  VRAM: ~37GB (Q8 mode) or ~77GB (BF16 mode)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits
1. **Shared CUDA context**: All services share GPU memory, enabling better utilization
2. **Single deployment**: One container to manage
3. **Reduced overhead**: Single PyTorch installation, shared libraries
4. **Lower latency**: Services communicate via localhost
5. **Flexibility**: Switch between llama.cpp (fast, quantized) and vLLM (quality, BF16) modes

## Pinned Commits for Reproducibility

All repositories are pinned to specific commit SHAs to ensure reproducible builds. The Dockerfile uses these exact commits rather than branch names.

| Repository | Branch | Commit SHA | Date | Notes |
|------------|--------|------------|------|-------|
| PyTorch | main | `32cb1dac896fe212d77073a4a53fee840c13442f` | 2025-12-26 | CUDA 13.1 support |
| torchaudio | main | `0764cfdedb769e63f3ab8b90bc06541a6a2c0b73` | 2025-12-20 | Compatible with PyTorch main |
| NeMo | main | `119593e7bf6e854a2136c3b48b804cae97657a0f` | 2026-01-26 | Includes Hindi tokenizer required by multilingual Magpie TTS |
| vLLM | main | `bb80f69bc98cbf062bf030cb11185f7ba526e28a` | 2025-12-21 | Supports Nemotron-H relu2_no_mul |
| llama.cpp | master | `c18428423018ed214c004e6ecaedb0cbdda06805` | 2025-12-24 | Tested on DGX Spark, avoids mxfp4 build issues |

### Updating Commits

To update to newer commits:
1. Look up the latest commit SHA from GitHub
2. Update the corresponding `ARG *_COMMIT=` line in `Dockerfile.unified`
3. Test the build to ensure compatibility
4. Update this table with the new SHA and date

## Build Strategy

### Multi-Stage Considerations

The build is intentionally NOT multi-stage because:
1. **Build time is acceptable**: 2-3 hours is a one-time cost
2. **Debug-ability**: Single stage makes troubleshooting easier
3. **Layer caching**: Docker caches layers effectively for iterative development
4. **Space savings limited**: Most size comes from PyTorch/CUDA, not build tools

### Build Phases

#### Phase 1: Base System (5 min)
- CUDA 13.1.0 devel base
- System dependencies (union of ASR + vLLM requirements)
- Python 3.12, uv package manager

#### Phase 2: CUDA Libraries (2 min)
- cuDNN libraries (from NGC PyTorch container)
- NCCL libraries (required for distributed/NeMo)
- Verify library installation

#### Phase 3: PyTorch Build (60-90 min)
- Clone PyTorch main branch (CUDA 13.1 compatibility)
- Build with `TORCH_CUDA_ARCH_LIST=12.1`
- Enable CUDA, cuDNN, MKLDNN, NCCL, distributed
- Install wheel, save for potential reinstallation
- Cleanup build artifacts

#### Phase 4: torchaudio Build (15-20 min)
- Clone torchaudio main branch
- Build with `USE_CUDA=1`
- Required for NeMo ASR
- Cleanup build artifacts

#### Phase 5: NeMo Installation (10-15 min)
- Clone NeMo main branch (specific commit for reproducibility)
- Install ASR + TTS dependencies
- Install NeMo in editable mode with ASR + TTS extras
- Apply NVRTC patch for dynamic GPU architecture detection

#### Phase 6: vLLM Build (30-45 min)
- Install vLLM dependencies (tokenizers, fastapi, etc.)
- Clone and build vLLM from source
- Reinstall our custom PyTorch (vLLM may replace it)
- Rebuild torchaudio (needed after PyTorch reinstall)
- Install and patch triton for sm_121a

#### Phase 7: llama.cpp Build (5-10 min)
- Clone llama.cpp
- Build with CUDA support, sm_121a architecture
- Install binaries (llama-server, llama-cli, llama-quantize)
- Cleanup build artifacts

#### Phase 8: Application (2 min)
- Install application dependencies (websockets, loguru, httpx)
- Install nemotron-speech package

### Total Build Time: ~2-3 hours

## Runtime Modes

The startup script supports flexible service selection. Any combination of LLM, ASR, and TTS can be enabled/disabled.

### Service Selection

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ENABLE_LLM` | `true`/`false` | `true` | Enable LLM server |
| `ENABLE_ASR` | `true`/`false` | `true` | Enable ASR server |
| `ENABLE_TTS` | `true`/`false` | `true` | Enable TTS server |
| `LLM_MODE` | `llamacpp-q8`, `llamacpp-q4`, `vllm` | `llamacpp-q8` | LLM backend mode |

### LLM Mode: llama.cpp Q8 (Default)

For Q8 quantized GGUF models - best balance of quality and VRAM:

```bash
docker run -d --name nemotron --runtime=nvidia --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/hf_cache:ro \
  -p 8000:8000 -p 8001:8001 -p 8080:8080 \
  -e LLAMA_MODEL=/hf_cache/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q8_0.gguf \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-unified:cuda13 \
  bash /workspace/scripts/start_unified.sh
```

**Configuration:**
- `LLM_MODE=llamacpp-q8` (default)
- `LLAMA_MODEL` - path to Q8 GGUF model file
- `LLAMA_PARALLEL=2` - number of parallel slots (required for two-slot alternation)
- VRAM: ~37GB total (~5GB ASR/TTS + ~32GB LLM)

### LLM Mode: llama.cpp Q4

For Q4 quantized GGUF models - lower VRAM, slightly reduced quality:

```bash
docker run -d --name nemotron --runtime=nvidia --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/hf_cache:ro \
  -p 8000:8000 -p 8001:8001 -p 8080:8080 \
  -e LLM_MODE=llamacpp-q4 \
  -e LLAMA_MODEL=/hf_cache/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q4_K_M.gguf \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-unified:cuda13 \
  bash /workspace/scripts/start_unified.sh
```

**Configuration:**
- `LLM_MODE=llamacpp-q4`
- `LLAMA_MODEL` - path to Q4 GGUF model file
- VRAM: ~21GB total (~5GB ASR/TTS + ~16GB LLM)

### LLM Mode: vLLM (Full Weights)

For BF16 full-precision inference - highest quality:

```bash
docker run -d --name nemotron --runtime=nvidia --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $(pwd)/models:/workspace/models:ro \
  -p 8000:8000 -p 8001:8001 -p 8080:8080 \
  -e LLM_MODE=vllm \
  -e VLLM_MODEL=/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  nemotron-unified:cuda13 \
  bash /workspace/scripts/start_unified.sh
```

**Configuration:**
- `LLM_MODE=vllm`
- `VLLM_MODEL` - path to HuggingFace model directory
- `VLLM_GPU_MEMORY_UTILIZATION=0.60` - leave room for ASR/TTS
- VRAM: ~77GB total (~5GB ASR/TTS + ~72GB LLM)

### Partial Service Examples

```bash
# LLM only (no ASR/TTS) - for standalone LLM server
ENABLE_ASR=false ENABLE_TTS=false bash /workspace/scripts/start_unified.sh

# ASR + TTS only (no LLM) - for speech services only
ENABLE_LLM=false bash /workspace/scripts/start_unified.sh

# ASR only
ENABLE_LLM=false ENABLE_TTS=false bash /workspace/scripts/start_unified.sh

# TTS only
ENABLE_LLM=false ENABLE_ASR=false bash /workspace/scripts/start_unified.sh
```

## Startup Script Design

### scripts/start_unified.sh (In-Container)

The in-container startup script manages service lifecycle with health-check polling.

**Features:**
1. Enable/disable individual services via `ENABLE_ASR`, `ENABLE_TTS`, `ENABLE_LLM`
2. Multiple LLM backends: `llamacpp-q8` (default), `llamacpp-q4`, `vllm`
3. Health-check polling (not fixed sleep) to detect service readiness
4. Service logs written to `/var/log/nemotron/{asr,tts,llm}.log`
5. Graceful shutdown on SIGTERM/SIGINT (prevents orphaned GPU processes)

### scripts/nemotron.sh (Host-Side)

The host-side wrapper manages the Docker container lifecycle.

**Commands:**
```bash
./scripts/nemotron.sh start [OPTIONS]   # Start container
./scripts/nemotron.sh stop              # Stop container
./scripts/nemotron.sh restart [OPTIONS] # Restart container
./scripts/nemotron.sh status            # Check service health
./scripts/nemotron.sh logs [SERVICE]    # View logs (asr, tts, llm, all)
./scripts/nemotron.sh shell             # Open shell in container
```

**Start Options:**
- `--mode MODE` - LLM mode: `llamacpp-q8`, `llamacpp-q4`, `vllm`
- `--model PATH` - Model path (optional; auto-detects from HuggingFace cache)
- `--no-asr`, `--no-tts`, `--no-llm` - Disable services
- `-f` / `--foreground` - Run in foreground instead of detached

### Key Design Decisions

1. **ASR/TTS First**: These services consume ~5GB VRAM and load quickly. Starting them first ensures they claim GPU memory before the LLM.

2. **GPU Warmup Before Ready**: Both ASR and TTS run warmup inference during startup to force CUDA kernel compilation and memory allocation. This ensures GPU memory is actually claimed before the LLM starts (PyTorch lazy initialization would otherwise delay allocation until first real request).

3. **Service-Specific Health Checks**:
   - **ASR (WebSocket-only)**: Waits for "GPU memory claimed" log message after warmup
   - **TTS (HTTP + WebSocket)**: Polls `/health` HTTP endpoint
   - **LLM (HTTP)**: Polls `/health` HTTP endpoint (both llama.cpp and vLLM support this)

4. **Per-Service Logs**: Each service logs to `/var/log/nemotron/{asr,tts,llm}.log`, enabling independent log viewing via `nemotron.sh logs <service>`.

5. **Environment Validation**: The script fails fast if required model paths aren't set, avoiding cryptic errors later.

6. **Graceful Shutdown**: The cleanup handler sends SIGTERM, waits 5 seconds, then SIGKILL if needed. This prevents orphaned processes holding GPU memory.

## Testing Plan

### 1. ASR Test

```bash
# Test ASR WebSocket endpoint
python tests/test_streaming_server.py
```

Expected: Transcription works with streaming audio.

### 2. TTS Test

```bash
# Test TTS WebSocket endpoint
python tests/test_magpie_tts.py
```

Expected: Audio generated from text input.

### 3. llama.cpp LLM Test

```bash
# Test llama.cpp completions endpoint
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am", "n_predict": 20}'
```

Expected: JSON response with generated text.

### 4. vLLM LLM Test

```bash
# Test vLLM OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: OpenAI-format JSON response.

### 5. Integration Test

```bash
# Full pipeline test
python tests/test_llm_tts_integration.py
```

Expected: LLM generates text, TTS converts to audio.

## Environment Variables Reference

### Service Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LLM` | `true` | Enable LLM server on port 8000 |
| `ENABLE_ASR` | `true` | Enable ASR server on port 8080 |
| `ENABLE_TTS` | `true` | Enable TTS server on port 8001 |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODE` | `llamacpp-q8` | LLM backend: `llamacpp-q8`, `llamacpp-q4`, or `vllm` |
| `LLAMA_MODEL` | - | Path to GGUF model file (required for llamacpp modes) |
| `LLAMA_PARALLEL` | `2` | Number of parallel llama.cpp slots |
| `LLAMA_CTX_SIZE` | `65536` | Context size for llama.cpp |
| `VLLM_MODEL` | - | Path to HuggingFace model directory (required for vllm mode) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | GPU memory fraction for vLLM |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_ACCESS_TOKEN` | - | HuggingFace token for gated models |
| `SERVICE_TIMEOUT` | `60` (llama.cpp) / `900` (vLLM) | Seconds to wait for each service health check |

## Model Storage Strategy

All models use the HuggingFace cache at `~/.cache/huggingface` which is mounted into the container at `/root/.cache/huggingface`. This provides:
- Automatic model downloads via HuggingFace Hub
- Shared cache between host and container
- Support for gated models via `HUGGINGFACE_ACCESS_TOKEN`

| Model | HuggingFace ID | Size | Mode |
|-------|----------------|------|------|
| Parakeet ASR | `nvidia/parakeet-tdt-0.6b` | ~2.4GB | Both |
| Magpie TTS | `nvidia/magpie-tts-multilingual-357m` | ~1.4GB | Both |
| Nemotron-3-Nano Q8 GGUF | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | ~32GB | llama.cpp |
| Nemotron-3-Nano BF16 | `nvidia/Nemotron-3-Nano-30B-A3B` | ~60GB | vLLM |

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.unified` | Combined container Dockerfile |
| `scripts/start_unified.sh` | In-container service manager |
| `scripts/nemotron.sh` | Host-side container manager |
| `docs/unified-container-plan.md` | This planning document |

## Build and Run Commands

### Build

```bash
# Build the unified container (2-3 hours)
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
```

### Run with nemotron.sh (Recommended)

The `scripts/nemotron.sh` wrapper manages the container lifecycle from the host. Models are auto-detected from the HuggingFace cache:

```bash
# Start with llama.cpp Q8 (default) - auto-detects model
./scripts/nemotron.sh start

# Start with llama.cpp Q4 - auto-detects model
./scripts/nemotron.sh start --mode llamacpp-q4

# Start with vLLM - uses nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 by default
./scripts/nemotron.sh start --mode vllm

# Start ASR + TTS only (no LLM)
./scripts/nemotron.sh start --no-llm

# View logs
./scripts/nemotron.sh logs         # All services
./scripts/nemotron.sh logs llm     # LLM only
./scripts/nemotron.sh logs asr     # ASR only

# Check status
./scripts/nemotron.sh status

# Stop container
./scripts/nemotron.sh stop
```

### Run with docker directly

For more control, you can run docker commands directly:

```bash
# llama.cpp mode
docker run -d --name nemotron --runtime=nvidia --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 -p 8001:8001 -p 8080:8080 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  -e LLAMA_MODEL=/root/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q8_0.gguf \
  nemotron-unified:cuda13 \
  bash /workspace/scripts/start_unified.sh

# vLLM mode
docker run -d --name nemotron --runtime=nvidia --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 -p 8001:8001 -p 8080:8080 \
  -e HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN \
  -e LLM_MODE=vllm \
  -e VLLM_MODEL=nvidia/Nemotron-3-Nano-30B-A3B \
  nemotron-unified:cuda13 \
  bash /workspace/scripts/start_unified.sh
```

## Migration from Two Containers

1. Stop existing containers:
   ```bash
   docker stop nemotron-asr llama-q4
   docker rm nemotron-asr llama-q4
   ```

2. Build unified container:
   ```bash
   docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
   ```

3. Update `.env`:
   ```bash
   # No URL changes needed - same ports
   USE_WEBSOCKET_TTS=true
   USE_CHUNKED_LLM=true
   ```

4. Start unified container:
   ```bash
   ./scripts/nemotron.sh start
   ```

5. Verify services are running:
   ```bash
   ./scripts/nemotron.sh status
   ```

6. Run the bot:
   ```bash
   uv run pipecat_bots/bot.py
   ```

## Rollback Plan

If the unified container has issues:

1. Keep the original Dockerfiles (`Dockerfile.asr-cuda13-build`, `Dockerfile.vllm-llamacpp`)
2. Revert to two-container architecture
3. Debug unified container issues offline

## Known Issues and Solutions

### 1. PyTorch Version Conflicts

**Problem**: vLLM installation may replace our custom PyTorch.

**Solution**: After vLLM install, reinstall our PyTorch wheel:
```dockerfile
RUN uv pip install --no-cache --reinstall /tmp/pytorch_wheel/torch*.whl
```

### 2. torchaudio Compatibility

**Problem**: torchaudio must match PyTorch version exactly.

**Solution**: Rebuild torchaudio after any PyTorch reinstallation.

### 3. GPU Memory Management

**Problem**: LLM may consume all GPU memory before ASR/TTS load.

**Solution**: Start ASR/TTS first, wait 10 seconds, then start LLM.

### 4. llama.cpp Slot Crashes

**Problem**: `GGML_ASSERT(!slot.is_processing())` on interruption.

**Solution**: Use `--parallel 2` and two-slot alternation (handled by `LlamaCppChunkedLLMService`).

### 5. NeMo NVRTC Errors

**Problem**: NVRTC compilation fails on sm_121.

**Solution**: Patch NeMo's `cuda_python_utils.py` for dynamic architecture detection.

### 6. libcurl Dependency for llama.cpp

**Problem**: llama.cpp server requires libcurl for HTTP.

**Solution**: Include `libcurl4-openssl-dev` in system dependencies.

### 7. torchaudio Rebuild After vLLM

**Problem**: vLLM may install a different PyTorch version, breaking torchaudio.

**Solution**: After vLLM install and PyTorch reinstall, rebuild torchaudio from source.

### 8. vLLM Nemotron-H Activation Compatibility

**Problem**: Newer vLLM commits (after Dec 21, 2025) break support for Nemotron-H's `relu2_no_mul` MoE activation function, causing `ValueError: Unsupported FusedMoe activation: relu2_no_mul`.

**Solution**: Use vLLM commit `bb80f69bc98cbf062bf030cb11185f7ba526e28a` (Dec 21, 2025) which matches the working `vllm:cuda13-full` container.

**Fixes Applied to Dockerfile.unified**:

1. **vLLM commit**: Changed from `5326c89803566a131c928f7fdd2100b75c981a42` (Dec 26) to `bb80f69bc98cbf062bf030cb11185f7ba526e28a` (Dec 21) - matching the working `vllm:cuda13-full` container

2. **Build environment**: Added `TORCH_CUDA_ARCH_LIST="12.1"` and `MAX_JOBS=8`

3. **Build dependencies**: Added `packaging`, `wheel`, `jinja2`

4. **Install mode**: Changed back to editable install (`-e .`)

5. **Cleanup**: Keep `/build/vllm` for editable install

### 9. vLLM DNS Resolution with Docker Bridge Networking

**Problem**: vLLM mode fails with DNS resolution errors when trying to access HuggingFace to verify model files. Docker's default bridge networking uses a Tailscale DNS server that can't resolve external hostnames.

**Solution**: Use `--network=host` for vLLM mode. The `nemotron.sh` script automatically uses host networking when `--mode vllm` is specified.

### 10. Workspace vllm Directory Shadowing

**Problem**: If there's a `vllm/` directory in the project root, mounting `/workspace` causes Python to import from that directory instead of the installed vLLM package, resulting in `ImportError: cannot import name 'SamplingParams' from 'vllm'`.

**Solution**: Rename any local `vllm/` directory to something else (e.g., `vllm_plugins/`). The directory was renamed in the project.

### 11. vLLM Long Startup Time

**Problem**: vLLM takes ~10-15 minutes to load the 58GB Nemotron-3-Nano-30B-A3B-BF16 model. The default 60-second `SERVICE_TIMEOUT` causes startup to fail.

**Solution**: The `start_unified.sh` script automatically sets `SERVICE_TIMEOUT=900` (15 minutes) when `LLM_MODE=vllm`.

## Additional Considerations

### Image Size

The unified container will be large (~25-30GB) due to:
- CUDA devel base: ~5GB
- PyTorch built from source: ~3GB
- vLLM + dependencies: ~2GB
- NeMo + dependencies: ~5GB
- Build tools and caches: ~5-10GB

This is acceptable for a development/deployment image. For production, consider:
- Removing build tools after builds complete
- Using `--squash` to reduce layers
- Multi-stage build (though limited benefit here)

### Security

The container runs services as root by default. For production:
- Create a non-root user
- Use Docker security options
- Consider using Kubernetes with security contexts

### Monitoring

For production deployments, consider:
- Health check endpoints for each service
- Prometheus metrics export
- Structured logging (JSON)
- Service discovery integration

### Model Management

Models can be:
1. **Mounted at runtime**: Best for flexibility, development
2. **Baked into image**: Best for immutable deployments
3. **Downloaded on startup**: Best for CI/CD (but slow startup)

The plan uses runtime mounting for maximum flexibility.

## Future Improvements

The following improvements are deferred but should be considered for production:

### High Priority

1. **ASR HTTP Health Check**: The ASR server is WebSocket-only, so the `nemotron.sh status` command shows "DOWN" even when ASR is running. Add an HTTP `/health` endpoint to the ASR server for consistent status checking across all services.

2. **vLLM Local Path Handling**: The `nemotron.sh` script doesn't properly mount local model directories for vLLM mode. Currently only HuggingFace model IDs work reliably.

3. **Log Persistence**: Logs at `/var/log/nemotron/` are lost when the container restarts. Consider mounting a host volume:
   ```bash
   -v "$PROJECT_DIR/logs:/var/log/nemotron"
   ```

4. **ARM64 Architecture Check**: Add explicit check in Dockerfile since it only works on ARM64 (DGX Spark):
   ```dockerfile
   RUN if [ "$(uname -m)" != "aarch64" ]; then \
         echo "ERROR: This Dockerfile is for ARM64 only" && exit 1; \
       fi
   ```

### Medium Priority

5. **Port Customization**: Allow configuring service ports via `nemotron.sh`:
   ```bash
   ./scripts/nemotron.sh start --port-llm 9000 --port-tts 9001 --port-asr 9080
   ```

6. **Pip Dependency Pinning**: Generate a lockfile or pin exact versions for reproducibility. Currently pip packages may drift over time.

7. **Service Auto-Restart**: Consider supervisord for per-service restart on crash (current behavior: any service exit stops the container).

### Low Priority

8. **Prometheus Metrics**: Add `/metrics` endpoints for monitoring (request latency, GPU memory, queue depth).

9. **Config File Support**: Replace environment variable sprawl with a YAML/TOML config file for complex deployments.

10. **Resource Limits**: Add CPU/memory limits to prevent container from starving other processes.

11. **Non-Root User**: Create a dedicated user for running services instead of root.

## Alternative: Supervisor/systemd

Instead of a shell script managing processes, consider:

### supervisord

```ini
[supervisord]
nodaemon=true

[program:asr]
command=python -m nemotron_speech.server --port 8080
autorestart=true

[program:tts]
command=python -m nemotron_speech.tts_server --port 8001
autorestart=true

[program:llm]
command=llama-server -m %(ENV_LLAMA_MODEL)s --host 0.0.0.0 ...
autorestart=true
```

Benefits:
- Automatic restart on crash
- Better logging control
- Standard process management

For this initial implementation, the shell script is simpler and sufficient.

## README Updates Required

When completing the migration, update the README.md with the following:

### Quick Start Section
- Replace two-container instructions with unified container:
  ```bash
  # Single unified container (replaces both nemotron-asr and llama-q4)
  export HUGGINGFACE_ACCESS_TOKEN=$(cat ~/.cache/huggingface/token)
  ./scripts/nemotron.sh start
  ```

### Architecture Diagram
- Update to show single container with all three services
- Update VRAM requirements (~37GB for Q8 mode)

### Container Management Section
- Replace `docker logs -f nemotron-asr` and `docker logs -f llama-q4` with:
  ```bash
  ./scripts/nemotron.sh logs        # All services
  ./scripts/nemotron.sh logs asr    # ASR only
  ./scripts/nemotron.sh logs tts    # TTS only
  ./scripts/nemotron.sh logs llm    # LLM only
  ```

### Building Containers Section
- Update build command:
  ```bash
  docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
  ```

### Environment Variables
- Add documentation for `HUGGINGFACE_ACCESS_TOKEN` requirement
- Document `ENABLE_ASR`, `ENABLE_TTS`, `ENABLE_LLM` options

### Troubleshooting Section
- Add notes about HuggingFace token requirement for TTS
- Add migration notes for users transitioning from two-container setup
