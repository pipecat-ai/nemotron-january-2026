"""
Streaming TTS inference for Magpie TTS model.

This module provides streaming inference capabilities for the Magpie TTS model,
yielding audio chunks as soon as they're available to minimize TTFB.

Two streaming approaches are available:

1. Sentence-level streaming (RECOMMENDED):
   - Splits text into sentences
   - Generates each sentence using fast batch mode (~0.27x RTF)
   - Streams sentences as they complete
   - TTFB: ~200ms (first sentence), smooth playback

2. Frame-by-frame streaming (DEPRECATED - too slow):
   - Generates tokens one at a time
   - Very slow RTF (~2.8x), causes audio gaps
   - Only useful for very short utterances

Usage:
    # Inside container with NeMo/PyTorch available
    from streaming_tts import SentenceStreamingTTS

    streamer = SentenceStreamingTTS(model)
    for audio_chunk in streamer.synthesize_streaming(
        text="Hello! How can I help you today?",
        language="en",
        speaker_index=2,  # aria
    ):
        # audio_chunk is bytes (int16 PCM, 22kHz, mono)
        websocket.send(audio_chunk)

Performance:
    - TTFB: ~150-250ms (first sentence generation)
    - RTF: ~0.27x (batch mode, faster than real-time)
    - No audio gaps between chunks
"""

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class StreamingConfig:
    """Configuration for streaming TTS inference."""

    # Number of frames to accumulate before decoding
    # 1 frame = 1024 samples = ~46.4ms at 22kHz
    chunk_size_frames: int = 10  # ~465ms chunks

    # Overlap frames for smooth boundaries with non-causal decoder
    overlap_frames: int = 2  # ~93ms overlap

    # Minimum frames before first audio output (aggressive for low TTFB)
    min_first_chunk_frames: int = 4  # ~185ms TTFB target

    # Enable crossfade at chunk boundaries
    use_crossfade: bool = True
    crossfade_samples: int = 512  # ~23ms crossfade

    # Maximum decoder steps (same as batch mode)
    max_decoder_steps: int = 500

    # Generation parameters
    temperature: float = 0.7
    topk: int = 80
    use_cfg: bool = True
    cfg_scale: float = 2.5


# Preset configurations for different use cases
STREAMING_PRESETS = {
    # Aggressive: lowest TTFB, may have subtle artifacts
    "aggressive": StreamingConfig(
        min_first_chunk_frames=4,   # ~185ms TTFB
        chunk_size_frames=8,        # ~370ms chunks
        overlap_frames=2,
    ),
    # Balanced: good TTFB with reliable quality
    "balanced": StreamingConfig(
        min_first_chunk_frames=6,   # ~280ms TTFB
        chunk_size_frames=10,       # ~465ms chunks
        overlap_frames=2,
    ),
    # Conservative: prioritize quality over TTFB
    "conservative": StreamingConfig(
        min_first_chunk_frames=8,   # ~370ms TTFB
        chunk_size_frames=12,       # ~560ms chunks
        overlap_frames=3,
    ),
}


class StreamingMagpieTTS:
    """Streaming inference wrapper for MagpieTTSModel."""

    SAMPLE_RATE = 22000
    SAMPLES_PER_FRAME = 1024
    # Samples to preserve at chunk boundaries for server-side crossfade (80ms)
    # Increased from 40ms to 80ms to provide smoother transitions when the
    # non-causal HiFi-GAN vocoder produces uncorrelated waveforms at boundaries
    PRESERVE_OVERLAP_SAMPLES = int(SAMPLE_RATE * 0.080)  # 1760 samples = 80ms


    def __init__(self, model, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming TTS with a loaded MagpieTTSModel.

        Args:
            model: Loaded MagpieTTSModel instance (on GPU)
            config: Streaming configuration
        """
        self.model = model
        self.config = config or StreamingConfig()

        # Verify model is ready
        if not hasattr(model, "do_tts"):
            raise ValueError("Model must be a MagpieTTSModel with do_tts method")

    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        speaker_index: int = 2,  # aria
        apply_tn: bool = False,
    ) -> Generator[bytes, None, None]:
        """
        Synthesize speech with streaming output.

        Yields audio chunks as soon as they're available, minimizing TTFB.

        Args:
            text: Text to synthesize
            language: Language code (en, es, de, fr, vi, it, zh)
            speaker_index: Speaker voice index (0-4)
            apply_tn: Apply text normalization

        Yields:
            bytes: PCM audio data (int16, mono, 22kHz)
        """
        # Import NeMo utilities (only available in container)
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

        cfg = self.config
        model = self.model

        with torch.no_grad():
            # Prepare batch (same as do_tts)
            batch = self._prepare_batch(text, language, speaker_index, apply_tn)

            # Initialize decoder state
            model.decoder.reset_cache(use_cache=model.use_kv_cache_for_inference)
            context_tensors = model.prepare_context_tensors(batch)

            # Initialize token generation
            text = context_tensors.text
            audio_codes_bos = torch.full(
                (text.size(0), model.num_audio_codebooks, model.frame_stacking_factor),
                model.audio_bos_id,
                device=text.device,
            ).long()

            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            # Prepare CFG if enabled
            if cfg.use_cfg:
                dummy_cond, dummy_cond_mask, dummy_add_dec_input, dummy_add_dec_mask, _ = (
                    model.prepare_dummy_cond_for_cfg(
                        context_tensors.cond,
                        context_tensors.cond_mask,
                        context_tensors.additional_decoder_input,
                        context_tensors.additional_decoder_mask,
                    )
                )

            # Streaming state
            all_predictions = []
            pending_tokens = []
            overlap_buffer = None
            end_indices = {}
            first_chunk_yielded = False

            # Main generation loop
            for idx in range(cfg.max_decoder_steps // model.frame_stacking_factor):
                # Generate next token(s)
                audio_codes_embedded, audio_codes_embedded_lens = model.embed_audio_tokens(
                    audio_tokens=audio_codes_input,
                    audio_tokens_lens=audio_codes_lens,
                )
                audio_codes_mask = get_mask_from_lengths(audio_codes_embedded_lens)

                if context_tensors.additional_decoder_input is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors.additional_decoder_input, audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat(
                        [context_tensors.additional_decoder_mask, audio_codes_mask], dim=1
                    )
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = audio_codes_mask

                # Forward pass with optional CFG
                if cfg.use_cfg:
                    batch_size = audio_codes_embedded.size(0)
                    cfg_cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
                    cfg_cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)
                    cfg_audio_codes_embedded = torch.cat(
                        [_audio_codes_embedded, _audio_codes_embedded], dim=0
                    )
                    cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)

                    combined_logits, _, dec_out = model.forward(
                        dec_input_embedded=cfg_audio_codes_embedded,
                        dec_input_mask=cfg_audio_codes_mask,
                        cond=cfg_cond,
                        cond_mask=cfg_cond_mask,
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )

                    cond_logits = combined_logits[:batch_size]
                    uncond_logits = combined_logits[batch_size:]
                    all_code_logits = (1 - cfg.cfg_scale) * uncond_logits + cfg.cfg_scale * cond_logits
                else:
                    all_code_logits, _, dec_out = model.forward(
                        dec_input_embedded=_audio_codes_embedded,
                        dec_input_mask=_audio_codes_mask,
                        cond=context_tensors.cond,
                        cond_mask=context_tensors.cond_mask,
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )
                    batch_size = audio_codes_embedded.size(0)

                # Sample next tokens
                forbid_eos = idx * model.frame_stacking_factor < 4
                all_code_logits_t = all_code_logits[:, -1, :]
                audio_codes_next = model.sample_codes_from_logits(
                    all_code_logits_t,
                    temperature=cfg.temperature,
                    topk=cfg.topk,
                    unfinished_items={},
                    finished_items={},
                    forbid_audio_eos=forbid_eos,
                )

                # Check for EOS
                all_codes_next_argmax = model.sample_codes_from_logits(
                    all_code_logits_t, temperature=0.01, topk=1,
                    unfinished_items={}, finished_items={}, forbid_audio_eos=forbid_eos,
                )

                # Import EOS detection enum
                from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod
                eos_method = EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        end_frame_index = model.detect_eos(
                            audio_codes_next[item_idx],
                            all_codes_next_argmax[item_idx],
                            eos_method,
                        )
                        if end_frame_index != float('inf'):
                            end_indices[item_idx] = idx * model.frame_stacking_factor + end_frame_index
                            logger.debug(f"Streaming: EOS detected at step {idx}, frame_stacking={model.frame_stacking_factor}, end_frame_index={end_frame_index}, global_eos_frame={end_indices[item_idx]}")

                # Accumulate predictions
                all_predictions.append(audio_codes_next)
                pending_tokens.append(audio_codes_next)
                audio_codes_input = torch.cat([audio_codes_input, audio_codes_next], dim=-1)
                audio_codes_lens = audio_codes_lens + model.frame_stacking_factor

                # Check if we should decode and yield a chunk
                total_pending_frames = len(pending_tokens) * model.frame_stacking_factor

                should_yield = False
                if not first_chunk_yielded:
                    # First chunk: yield once we have enough for min TTFB
                    if total_pending_frames >= cfg.min_first_chunk_frames:
                        should_yield = True
                        first_chunk_yielded = True
                else:
                    # Subsequent chunks: yield at chunk_size intervals
                    if total_pending_frames >= cfg.chunk_size_frames:
                        should_yield = True

                # Also yield on EOS
                if len(end_indices) > 0:
                    should_yield = True

                if should_yield and pending_tokens:
                    # Decode accumulated tokens
                    pending_codes = torch.cat(pending_tokens, dim=-1)

                    # Include overlap buffer for context
                    if overlap_buffer is not None:
                        decode_codes = torch.cat([overlap_buffer, pending_codes], dim=-1)
                        overlap_samples = cfg.overlap_frames * self.SAMPLES_PER_FRAME
                        overlap_frames = cfg.overlap_frames
                    else:
                        decode_codes = pending_codes
                        overlap_samples = 0
                        overlap_frames = 0

                    # Determine decode length - truncate at EOS if detected
                    # CRITICAL: Must actually slice the tensor, not just set length!
                    # HiFi-GAN decoder processes ALL tokens in tensor regardless of tokens_len parameter.
                    # In batch mode this works because generation stops at EOS so tensor only has valid tokens.
                    # In streaming we must explicitly slice to match batch behavior.
                    if len(end_indices) > 0 and 0 in end_indices:
                        # Calculate EOS frame position relative to current decode_codes
                        # end_indices[0] is global frame index, we need local index
                        total_frames_before = (len(all_predictions) - len(pending_tokens)) * model.frame_stacking_factor
                        eos_global_frame = end_indices[0]
                        eos_local_frame = eos_global_frame - total_frames_before + overlap_frames
                        # Truncate at EOS frame - do NOT include the EOS frame itself
                        # This matches batch mode: predicted_lens = [end_indices.get(idx, max_decoder_steps)]
                        # where the EOS index is used directly as length (exclusive end)
                        truncated_len = min(eos_local_frame, decode_codes.size(-1))
                        logger.debug(f"Streaming: Truncating at EOS, global_frame={eos_global_frame}, local_frame={eos_local_frame}, truncated_len={truncated_len}, decode_codes_len={decode_codes.size(-1)}")
                        # Actually slice the tensor - HiFi-GAN processes ALL input tokens
                        decode_codes = decode_codes[..., :truncated_len]
                        decode_lens = torch.tensor([truncated_len], device=decode_codes.device).long()
                    else:
                        decode_lens = torch.tensor([decode_codes.size(-1)], device=decode_codes.device).long()

                    audio, audio_len, _ = model.codes_to_audio(decode_codes, decode_lens)

                    # Extract new audio (skip overlap region, but preserve some for server-side blending)
                    # The preserved overlap represents the same time period as the previous chunk's tail,
                    # enabling COLA-compliant overlap-add at the server with zero audio loss
                    if overlap_samples > 0:
                        preserve = min(overlap_samples, self.PRESERVE_OVERLAP_SAMPLES)
                        new_audio = audio[..., overlap_samples - preserve:]
                        logger.debug(f"Streaming chunk: audio_len={audio.shape[-1]}, overlap_samples={overlap_samples}, preserve={preserve}, new_audio_len={new_audio.shape[-1]}")
                    else:
                        new_audio = audio
                        logger.debug(f"Streaming chunk (first): audio_len={audio.shape[-1]}, no overlap")

                    # Update overlap buffer for next chunk
                    if pending_codes.size(-1) >= cfg.overlap_frames:
                        overlap_buffer = pending_codes[..., -cfg.overlap_frames:]
                    else:
                        overlap_buffer = pending_codes.clone()

                    # Convert to bytes and yield
                    audio_bytes = self._audio_to_bytes(new_audio)
                    yield audio_bytes

                    # Clear pending
                    pending_tokens = []

                # Check termination
                if len(end_indices) == text.size(0) and len(all_predictions) >= 4:
                    logger.debug(f"Streaming: All EOS detected at step {idx}, ending generation")
                    break

            # Yield any remaining audio
            if pending_tokens:
                pending_codes = torch.cat(pending_tokens, dim=-1)
                if overlap_buffer is not None:
                    decode_codes = torch.cat([overlap_buffer, pending_codes], dim=-1)
                    overlap_samples = cfg.overlap_frames * self.SAMPLES_PER_FRAME
                    overlap_frames = cfg.overlap_frames
                else:
                    decode_codes = pending_codes
                    overlap_samples = 0
                    overlap_frames = 0

                # Truncate at EOS if detected (same as main loop)
                # CRITICAL: Must actually slice tensor - HiFi-GAN ignores tokens_len for computation
                if len(end_indices) > 0 and 0 in end_indices:
                    total_frames_before = (len(all_predictions) - len(pending_tokens)) * model.frame_stacking_factor
                    eos_global_frame = end_indices[0]
                    eos_local_frame = eos_global_frame - total_frames_before + overlap_frames
                    # Do NOT include EOS frame itself (matches batch mode)
                    truncated_len = min(eos_local_frame, decode_codes.size(-1))
                    logger.debug(f"Streaming final: Truncating at EOS, truncated_len={truncated_len}, decode_codes_len={decode_codes.size(-1)}")
                    # Actually slice the tensor
                    decode_codes = decode_codes[..., :truncated_len]
                    decode_lens = torch.tensor([truncated_len], device=decode_codes.device).long()
                else:
                    decode_lens = torch.tensor([decode_codes.size(-1)], device=decode_codes.device).long()

                audio, _, _ = model.codes_to_audio(decode_codes, decode_lens)

                # Preserve overlap for server-side blending (same as main loop)
                if overlap_samples > 0:
                    preserve = min(overlap_samples, self.PRESERVE_OVERLAP_SAMPLES)
                    new_audio = audio[..., overlap_samples - preserve:]
                    logger.debug(f"Streaming final chunk: audio_len={audio.shape[-1]}, overlap_samples={overlap_samples}, preserve={preserve}, new_audio_len={new_audio.shape[-1]}")
                else:
                    new_audio = audio
                    logger.debug(f"Streaming final chunk (first): audio_len={audio.shape[-1]}, no overlap")

                audio_bytes = self._audio_to_bytes(new_audio)
                logger.debug(f"Streaming complete: yielding final {len(audio_bytes)} bytes")
                yield audio_bytes

    def _prepare_batch(
        self, text: str, language: str, speaker_index: int, apply_tn: bool
    ) -> dict:
        """Prepare input batch (mirrors do_tts logic)."""
        model = self.model

        # Apply text normalization if requested
        if apply_tn:
            normalized_text = model._get_normalized_text(transcript=text, language=language)
        else:
            normalized_text = text

        # Determine tokenizer
        tokenizer_name = None
        available_tokenizers = list(model.tokenizer.tokenizers.keys())
        language_tokenizer_map = {
            "en": ["english_phoneme", "english"],
            "de": ["german_phoneme", "german"],
            "es": ["spanish_phoneme", "spanish"],
            "fr": ["french_chartokenizer", "french"],
            "it": ["italian_phoneme", "italian"],
            "vi": ["vietnamese_phoneme", "vietnamese"],
            "zh": ["mandarin_phoneme", "mandarin", "chinese"],
        }

        if language in language_tokenizer_map:
            for candidate in language_tokenizer_map[language]:
                if candidate in available_tokenizers:
                    tokenizer_name = candidate
                    break

        if tokenizer_name is None:
            tokenizer_name = available_tokenizers[0]

        # Tokenize
        tokens = model.tokenizer.encode(text=normalized_text, tokenizer_name=tokenizer_name)
        tokens = tokens + [model.eos_id]
        text_tensor = torch.tensor([tokens], device=model.device, dtype=torch.long)
        text_lens = torch.tensor([len(tokens)], device=model.device, dtype=torch.long)

        return {
            'text': text_tensor,
            'text_lens': text_lens,
            'speaker_indices': speaker_index,
        }

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM bytes."""
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        # Handle empty audio
        if audio_np.size == 0:
            return b""

        # Normalize and convert to int16
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _apply_crossfade(
        self, prev_tail: torch.Tensor, new_audio: torch.Tensor, crossfade_samples: int
    ) -> torch.Tensor:
        """Apply crossfade between previous chunk tail and new audio."""
        if prev_tail.size(-1) < crossfade_samples or new_audio.size(-1) < crossfade_samples:
            return new_audio

        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=new_audio.device)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=new_audio.device)

        # Apply crossfade
        new_audio_start = new_audio[..., :crossfade_samples]
        prev_end = prev_tail[..., -crossfade_samples:]

        crossfaded = prev_end * fade_out + new_audio_start * fade_in

        # Replace start of new audio with crossfaded portion
        result = new_audio.clone()
        result[..., :crossfade_samples] = crossfaded

        return result


class SentenceStreamingTTS:
    """Sentence-level streaming TTS using fast batch mode.

    This approach splits text into sentences and generates each using the
    efficient batch mode (do_tts), streaming sentences as they complete.

    This achieves:
    - Low TTFB (~150-250ms for first sentence)
    - No audio gaps (batch mode is faster than real-time)
    - High quality (no frame-by-frame artifacts)
    """

    SAMPLE_RATE = 22000

    def __init__(self, model):
        """Initialize with a loaded MagpieTTSModel."""
        self.model = model

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming.

        Handles common sentence endings while preserving punctuation.
        """
        import re

        # Split on sentence-ending punctuation followed by space or end
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Filter empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentence breaks found, return the whole text
        if not sentences:
            return [text.strip()] if text.strip() else []

        return sentences

    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        speaker_index: int = 2,
        apply_tn: bool = False,
    ) -> Generator[bytes, None, None]:
        """Synthesize speech with sentence-level streaming.

        Splits text into sentences and generates each using fast batch mode.
        Yields audio as soon as each sentence is ready.

        Args:
            text: Text to synthesize
            language: Language code (en, es, de, fr, vi, it, zh)
            speaker_index: Speaker voice index (0-4)
            apply_tn: Apply text normalization

        Yields:
            bytes: PCM audio data (int16, mono, 22kHz)
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return

        with torch.no_grad():
            for sentence in sentences:
                # Generate using fast batch mode
                audio, audio_len = self.model.do_tts(
                    sentence,
                    language=language,
                    speaker_index=speaker_index,
                    apply_TN=apply_tn,
                )

                # Convert to bytes
                audio_bytes = self._audio_to_bytes(audio)
                yield audio_bytes

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM bytes."""
        audio_np = audio.cpu().float().numpy()

        # Handle tensor shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze()

        # Handle empty audio
        if audio_np.size == 0:
            return b""

        # Normalize and convert to int16
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()


# Simple test
if __name__ == "__main__":
    print("Streaming TTS module loaded.")
    print("Run inside the nemotron-asr container with NeMo available.")
    print()
    print("RECOMMENDED: Use SentenceStreamingTTS for low-latency streaming:")
    print("  from nemo.collections.tts.models import MagpieTTSModel")
    print("  model = MagpieTTSModel.from_pretrained('nvidia/magpie_tts_multilingual_357m')")
    print("  model = model.cuda().eval()")
    print()
    print("  from streaming_tts import SentenceStreamingTTS")
    print("  streamer = SentenceStreamingTTS(model)")
    print("  for chunk in streamer.synthesize_streaming('Hello! How are you?'):")
    print("      print(f'Received {len(chunk)} bytes')")
