"""
stt.py — Speech-to-Text using Moonshine (Useful Sensors)
==========================================================
Moonshine is a tiny, fast, fully offline STT model.
Install:
    pip install useful-moonshine sounddevice numpy

Models available (all offline):
    "moonshine/tiny"   — fastest, ~50MB
    "moonshine/base"   — more accurate, ~100MB

Usage:
    stt = SpeechInput()
    text = stt.listen()   # blocks until speech is captured
    print(text)
"""

from __future__ import annotations

import sys
import threading
import numpy as np

SAMPLE_RATE    = 16000   # Moonshine requires 16kHz mono
SILENCE_THRESH = 0.01    # RMS below this = silence
SILENCE_SEC    = 1.5     # seconds of silence to auto-stop recording
MAX_SEC        = 15      # hard cap per utterance
MOONSHINE_MODEL = "moonshine/tiny"


class SpeechInput:
    """
    Records from microphone until silence, then transcribes with Moonshine.
    Thread-safe. Re-usable across multiple calls.
    """

    def __init__(self, model: str = MOONSHINE_MODEL):
        self._model_name = model
        self._model      = None   # lazy load on first use
        self._lock       = threading.Lock()

    # ------------------------------------------------------------------
    def _load_model(self):
        """Lazy-load Moonshine on first listen() call."""
        if self._model is not None:
            return
        try:
            import moonshine
            self._moonshine = moonshine
            self._model = moonshine.load(self._model_name)
        except ImportError:
            raise ImportError(
                "Moonshine not installed.\n"
                "Run: pip install useful-moonshine sounddevice numpy"
            )

    # ------------------------------------------------------------------
    def listen(self, prompt: str = "🎤  Listening …") -> str | None:
        """
        Record audio from the default microphone.
        Stops automatically after SILENCE_SEC seconds of silence
        or MAX_SEC total.

        Returns transcribed text string, or None on failure.
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
        silence_limit = int(SILENCE_SEC * SAMPLE_RATE / 512)  # in blocks
        max_blocks    = int(MAX_SEC    * SAMPLE_RATE / 512)
        block_count   = 0
        has_speech    = False

        def _callback(indata, frame_count, time_info, status):
            nonlocal silence_count, block_count, has_speech
            block_count += 1
            audio_block  = indata[:, 0].copy()   # mono
            rms          = float(np.sqrt(np.mean(audio_block ** 2)))

            frames.append(audio_block)

            if rms > SILENCE_THRESH:
                has_speech    = True
                silence_count = 0
            else:
                silence_count += 1

        # Open stream and record
        stop_event = threading.Event()

        def _record():
            with sd.InputStream(
                samplerate  = SAMPLE_RATE,
                channels    = 1,
                dtype       = "float32",
                blocksize   = 512,
                callback    = _callback,
            ):
                while not stop_event.is_set():
                    sd.sleep(50)  # check every 50ms

        record_thread = threading.Thread(target=_record, daemon=True)
        record_thread.start()

        # Wait until silence-stop or max duration
        import time
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

        # Concatenate and transcribe
        audio = np.concatenate(frames, axis=0)
        print("  ✓ Transcribing …")

        try:
            with self._lock:
                result = self._moonshine.transcribe(self._model, audio)
            # Moonshine returns a list of strings
            text = result[0].strip() if isinstance(result, list) else str(result).strip()
            print(f"  ✓ Heard: \"{text}\"")
            return text
        except Exception as e:
            print(f"  ✗ Transcription error: {e}")
            return None

    # ------------------------------------------------------------------
    def close(self):
        """Release model resources."""
        self._model = None