import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASR_MODELS_DIR = MODELS_DIR / "asr"
TRANSLATION_MODELS_DIR = MODELS_DIR / "translation"
TTS_MODELS_DIR = MODELS_DIR / "tts"
EMOTION_MODELS_DIR = MODELS_DIR / "emotion"
SAMPLE_AUDIO_DIR = MODELS_DIR / "sample_audio"
DATA_DIR = BASE_DIR / "backend" / "data"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"

for d in [MODELS_DIR, ASR_MODELS_DIR, TRANSLATION_MODELS_DIR, TTS_MODELS_DIR, EMOTION_MODELS_DIR, SAMPLE_AUDIO_DIR, DATA_DIR, EMOTION_SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Strict Offline Enforcement Flag
OFFLINE_MODE_STRICT = True
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
