from typing import Dict, Any, Optional
from backend.services.model_manager import modelManager

class EmotionService:
    """
    Offline Speech Emotion Recognition Service.
    Delegates to the currently active EmotionModelAdapter (emotion2vec+ Base).
    Zero cloud API reliance.
    """
    def analyze_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        adapter = modelManager.get_adapter("emotion")
        if not adapter or not adapter.is_loaded():
            raise RuntimeError("Emotion recognition model not loaded")

        return adapter.run_inference(audio_data, sample_rate=sample_rate)

    def get_supported_emotions(self) -> list:
        adapter = modelManager.get_adapter("emotion")
        if adapter:
            return adapter.get_supported_emotions()
        return ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

emotionService = EmotionService()
