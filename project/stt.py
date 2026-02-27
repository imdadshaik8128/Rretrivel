"""
stt.py — Speech-to-Text using OpenAI Whisper (tiny, offline)
==============================================================
Install:
    pip install openai-whisper sounddevice numpy
    # Windows: choco install ffmpeg  (or grab from ffmpeg.org)

Models (all offline after first download):
    "tiny"   — ~150MB, fastest
    "base"   — ~290MB, more accurate
    "small"  — ~970MB, even better
"""

from __future__ import annotations

import threading
import time
import numpy as np

SAMPLE_RATE    = 16000   # Whisper requires 16kHz mono
SILENCE_THRESH = 0.01    # RMS below this = silence
SILENCE_SEC    = 1.5     # seconds of silence to auto-stop
MAX_SEC        = 15      # hard cap per utterance
WHISPER_MODEL  = "tiny"  # change to "base" or "small" for better accuracy


class SpeechInput:
    """
    Records from microphone until silence, then transcribes with Whisper.
    Lazy-loads the model on first listen() call.
    """

    def __init__(self, model: str = WHISPER_MODEL):
        self._model_name = model
        self._model      = None
        self._lock       = threading.Lock()

    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            print(f"  ⏳  Loading Whisper '{self._model_name}' model (downloads once) …")
            self._model = whisper.load_model(self._model_name)
            print(f"  ✓  Whisper '{self._model_name}' ready.\n")
        except ImportError:
            raise ImportError(
                "Whisper not installed.\n"
                "Run: pip install openai-whisper sounddevice numpy"
            )

    # ------------------------------------------------------------------
    def listen(self, prompt: str = "🎤  Listening …") -> str | None:
        """
        Record from default microphone.
        Stops after SILENCE_SEC of silence or MAX_SEC total.
        Returns transcribed text or None if nothing heard.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed.\n"
                "Run: pip install sounddevice"
            )

        self._load_model()

        print(f"\n  {prompt}  (speak now — silence stops recording)\n")

        frames        = []
        silence_count = 0
        silence_limit = int(SILENCE_SEC * SAMPLE_RATE / 512)
        max_blocks    = int(MAX_SEC    * SAMPLE_RATE / 512)
        block_count   = 0
        has_speech    = False

        def _callback(indata, frame_count, time_info, status):
            nonlocal silence_count, block_count, has_speech
            block_count += 1
            block = indata[:, 0].copy()
            rms   = float(np.sqrt(np.mean(block ** 2)))
            frames.append(block)

            if rms > SILENCE_THRESH:
                has_speech    = True
                silence_count = 0
            else:
                silence_count += 1

        stop_event = threading.Event()

        def _record():
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = 1,
                dtype      = "float32",
                blocksize  = 512,
                callback   = _callback,
            ):
                while not stop_event.is_set():
                    sd.sleep(50)

        record_thread = threading.Thread(target=_record, daemon=True)
        record_thread.start()

        # Wait until silence-stop or max duration
        while True:
            time.sleep(0.05)
            if block_count >= max_blocks:
                print("  (max duration reached)")
                break
            if has_speech and silence_count >= silence_limit:
                break

        stop_event.set()
        record_thread.join(timeout=2)

        if not frames or not has_speech:
            print("  (no speech detected)")
            return None

        # Concatenate frames into single float32 array
        audio = np.concatenate(frames, axis=0).astype(np.float32)

        print("  ✓  Transcribing …")
        try:
            import whisper
            with self._lock:
                # fp16=False for CPU (Windows has no CUDA by default)
                result = self._model.transcribe(
                    audio,
                    fp16=False,
                    language="en",         # force English — faster, no language detect
                    task="transcribe",
                )
            text = result["text"].strip()
            if not text:
                print("  (whisper returned empty transcript)")
                return None
            print(f"  ✓  Heard: \"{text}\"")
            return text
        except Exception as e:
            print(f"  ✗  Transcription error: {e}")
            return None

    # ------------------------------------------------------------------
    def close(self):
        self._model = None