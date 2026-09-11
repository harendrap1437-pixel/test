import io
import wave
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from backend.config import SAMPLE_AUDIO_DIR

class AudioService:
    """
    Audio ingestion, format validation, duration measurement, and sample generation.
    Operates strictly in-memory or on local disk.
    """
    def validate_and_get_duration(self, audio_bytes: bytes, fallback_sr: int = 16000) -> float:
        """Returns exact audio duration in seconds."""
        try:
            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    return max(0.1, frames / float(rate))
        except Exception:
            # Raw 16-bit PCM fallback
            num_samples = len(audio_bytes) / 2
            return max(0.1, num_samples / float(fallback_sr))

    def create_sample_wav(self, duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
        """Generates a test WAV audio file for benchmarking."""
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        # Synthetic speech-like formant frequencies (150Hz + 600Hz + 1400Hz)
        waveform = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 600 * t) + 0.2 * np.sin(2 * np.pi * 1400 * t)
        
        # Apply amplitude envelope
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t / duration_sec))
        samples = (waveform * envelope * 32767).astype(np.int16)

        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())
        return bio.getvalue()

    def get_sample_audio_files(self) -> list:
        files = []
        for p in SAMPLE_AUDIO_DIR.glob("*.wav"):
            dur = self.validate_and_get_duration(p.read_bytes())
            files.append({
                "filename": p.name,
                "filepath": str(p),
                "duration_sec": round(dur, 2),
                "size_kb": round(p.stat().st_size / 1024, 2)
            })
        return files

audioService = AudioService()
