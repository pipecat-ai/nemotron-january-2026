#!/usr/bin/env python3
#
# Pipecat bot with interleaved streaming for lowest latency.
#
# Uses buffered LLM (sentence-boundary streaming) with adaptive WebSocket TTS.
# Single-slot operation achieves 100% KV cache reuse across turns.
# SmartTurn analyzer for responsive turn-taking.
#
# Environment variables:
#   NVIDIA_ASR_URL        ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_LLAMA_CPP_URL  llama.cpp API URL (default: http://localhost:8000)
#   NVIDIA_TTS_URL        Magpie TTS server URL (default: http://localhost:8001)
#   ENABLE_RECORDING      Enable audio recording (default: false)
#
# Usage:
#   uv run pipecat_bots/bot_interleaved_streaming.py
#   uv run pipecat_bots/bot_interleaved_streaming.py -t daily
#   uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc
#

import asyncio
import os
import time
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMMessagesFrame, LLMRunFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from magpie_websocket_tts import MagpieWebSocketTTSService
from llama_cpp_buffered_llm import LlamaCppBufferedLLMService
from v2v_metrics import V2VMetricsProcessor


class ContextTimingWrapper(FrameProcessor):
    """Log when LLMMessagesFrame passes through for V2V timing investigation."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            logger.debug(f"ContextTiming: LLMMessagesFrame at {time.time():.3f}")

        await self.push_frame(frame, direction)


load_dotenv(override=True)

# Configuration from environment
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLAMA_CPP_URL = os.getenv("NVIDIA_LLAMA_CPP_URL", "http://localhost:8000")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# Audio recording configuration
ENABLE_RECORDING = os.getenv("ENABLE_RECORDING", "false").lower() == "true"
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"

# VAD configuration - used by both VAD analyzer and V2V metrics
VAD_STOP_SECS = 0.2


def ensure_recordings_dir() -> Path:
    """Create recordings directory if it doesn't exist."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return RECORDINGS_DIR


async def save_audio_file(audio: bytes, sample_rate: int, num_channels: int, filepath: Path):
    """Save audio bytes to WAV file using asyncio.to_thread for non-blocking I/O."""

    def _write_wav():
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

        with open(filepath, "wb") as f:
            f.write(buffer.getvalue())

    try:
        await asyncio.to_thread(_write_wav)
        logger.info(f"Saved recording: {filepath} ({len(audio)} bytes, {sample_rate}Hz, {num_channels}ch)")
    except Exception as e:
        logger.error(f"Failed to save recording to {filepath}: {e}")

# Transport configurations with VAD and SmartTurn analyzer
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting interleaved streaming bot")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM URL: {NVIDIA_LLAMA_CPP_URL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL}")
    logger.info(f"  Transport: {type(transport).__name__}")
    logger.info(f"  Recording: {'enabled' if ENABLE_RECORDING else 'disabled'}")
    logger.info(f"  VAD stop_secs: {VAD_STOP_SECS}s")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # WebSocket Magpie TTS with adaptive mode
    # Adaptive mode: streaming for first segment (~370ms TTFB), batch for subsequent (quality)
    tts = MagpieWebSocketTTSService(
        server_url=NVIDIA_TTS_URL,
        voice="sofia",
        language="en",
        params=MagpieWebSocketTTSService.InputParams(
            language="en",
            streaming_preset="conservative",
            use_adaptive_mode=True,
        ),
    )
    logger.info("Using WebSocket Magpie TTS (adaptive mode)")

    # Voice-to-voice response time metrics
    v2v_metrics = V2VMetricsProcessor(vad_stop_secs=VAD_STOP_SECS)

    # Audio recording - stereo: user (left), bot (right)
    # Only create if recording is enabled
    audiobuffer = AudioBufferProcessor(num_channels=2) if ENABLE_RECORDING else None

    if audiobuffer:

        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
            """Save combined conversation audio when recording completes."""
            if len(audio) == 0:
                logger.warning("No audio data to save")
                return

            ensure_recordings_dir()
            # Use microseconds for filename uniqueness
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            filepath = RECORDINGS_DIR / f"{timestamp}.wav"
            await save_audio_file(audio, sample_rate, num_channels, filepath)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant running on an NVIDIA DGX Spark. "
                "You are built with Nemotron Three Nano, a large language model developed by NVIDIA. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters. Use only simple, plain text sentences. "
                "Always punctuate your responses using standard sentence punctuation: commas, periods, question marks, exclamation points, etc. "
                "Always spell out numbers as words. "
            ),
        },
        {
            "role": "user",
            "content": "Say hello and ask how you can help.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # LLM service - buffered mode (single slot, 100% KV cache reuse)
    llm = LlamaCppBufferedLLMService(
        llama_url=NVIDIA_LLAMA_CPP_URL,
        params=LlamaCppBufferedLLMService.InputParams(
            first_segment_max_tokens=24,
            first_segment_hard_max_tokens=24,
            segment_max_tokens=32,
            segment_hard_max_tokens=96,
        ),
    )
    logger.info("Using LlamaCppBufferedLLMService (single-slot, 100% cache)")

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Context timing wrapper for V2V latency investigation
    context_timing = ContextTimingWrapper()

    # Build pipeline processors
    pipeline_processors = [
        transport.input(),
        rtvi,
        stt,
        context_aggregator.user(),
        context_timing,  # Log when LLMMessagesFrame passes through
        llm,
        tts,
        v2v_metrics,
        transport.output(),
    ]

    # Add audio buffer if recording is enabled
    if audiobuffer:
        pipeline_processors.append(audiobuffer)

    pipeline_processors.append(context_aggregator.assistant())

    pipeline = Pipeline(pipeline_processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        if audiobuffer:
            await audiobuffer.start_recording()
            logger.info("Recording started")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
