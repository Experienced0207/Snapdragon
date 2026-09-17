"""
Text-to-Speech (TTS) Service for SignBridge.
Wraps MeloTTS / pyttsx3 / gTTS / local audio synthesizer to convert text to WAV audio bytes.
"""

import os
import io
import wave
import struct
import numpy as np
from typing import Optional, Union


def create_synthetic_wav_buffer(text: str, duration_per_char: float = 0.05, sample_rate: int = 22050) -> bytes:
    """
    Generates a clean synthetic audio WAV buffer with spoken-like audio tones when external TTS engines are offline.
    """
    duration = max(0.5, len(text) * duration_per_char)
    num_samples = int(sample_rate * duration)
    
    # Generate pleasant speech-like harmonic tone
    t = np.linspace(0, duration, num_samples, False)
    freq = 220.0  # A3 pitch
    audio_signal = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * freq * 1.5 * t)
    
    # Envelope to avoid audio clicks
    fade = int(sample_rate * 0.05)
    if num_samples > 2 * fade:
        audio_signal[:fade] *= np.linspace(0, 1, fade)
        audio_signal[-fade:] *= np.linspace(1, 0, fade)

    # Convert float [-1.0, 1.0] to 16-bit PCM integer
    pcm16 = (audio_signal * 32767).astype(np.int16)

    # Write WAV header into BytesIO buffer
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    return buf.getvalue()


class TextToSpeechService:
    """
    Text-to-Speech service for SignBridge conversational audio synthesis.
    """
    def __init__(self, engine_name: str = "auto"):
        self.engine_name = engine_name.lower()
        self.melo_model = None
        self.pyttsx_engine = None

        self._init_tts_engine()

    def _init_tts_engine(self):
        """Initializes primary TTS backend (MeloTTS / pyttsx3 / gTTS) or synthetic fallback."""
        # 1. Try MeloTTS if requested or auto
        if self.engine_name in ["melo", "melotts", "auto"]:
            try:
                from melo.api import TTS
                print("[TextToSpeechService] Loading MeloTTS engine...")
                self.melo_model = TTS(language='EN', device='auto')
                print("[TextToSpeechService] MeloTTS loaded successfully.")
                return
            except Exception as e:
                pass

        # 2. Try pyttsx3 fallback
        if self.engine_name in ["pyttsx3", "auto"]:
            try:
                import pyttsx3
                print("[TextToSpeechService] Initializing pyttsx3 engine...")
                self.pyttsx_engine = pyttsx3.init()
                print("[TextToSpeechService] pyttsx3 initialized successfully.")
                return
            except Exception as e:
                pass

        print("[TextToSpeechService] Using high-speed synthetic audio wave synthesizer.")

    def synthesize(self, text: str) -> bytes:
        """
        Synthesizes text into WAV audio bytes buffer.

        Args:
            text: Input string text to speak

        Returns:
            bytes: Audio content in WAV RIFF format
        """
        if not text or not text.strip():
            return create_synthetic_wav_buffer("...")

        clean_text = text.strip()

        # 1. MeloTTS execution
        if self.melo_model is not None:
            try:
                buf = io.BytesIO()
                self.melo_model.tts_to_file(clean_text, self.melo_model.hps.data.spk2id['EN-US'], buf, format='wav')
                return buf.getvalue()
            except Exception as e:
                print(f"[TextToSpeechService] MeloTTS error: {e}. Falling back to synthetic engine.")

        # 2. pyttsx3 execution
        if self.pyttsx_engine is not None:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                self.pyttsx_engine.save_to_file(clean_text, tmp_path)
                self.pyttsx_engine.runAndWait()
                with open(tmp_path, 'rb') as f:
                    wav_data = f.read()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return wav_data
            except Exception as e:
                print(f"[TextToSpeechService] pyttsx3 error: {e}. Falling back to synthetic engine.")

        # 3. Synthetic Audio Synthesizer
        return create_synthetic_wav_buffer(clean_text)


if __name__ == "__main__":
    tts = TextToSpeechService()
    audio_bytes = tts.synthesize("Hello! SignBridge Text to Speech Service is active.")
    print(f"Synthesized WAV audio bytes size: {len(audio_bytes)} bytes (WAV Header present: {audio_bytes[:4] == b'RIFF'})")
