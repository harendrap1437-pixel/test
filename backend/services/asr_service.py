from typing import Dict, Any, Optional
from backend.services.model_manager import modelManager

class ASRService:
    """
    Offline ASR Service. Delegates to the currently active ASRModelAdapter.
    Zero cloud STT fallback; zero browser SpeechRecognition.
    """
    def transcribe(self, audio_data: bytes, sample_rate: int = 16000, language: str = "hi") -> Dict[str, Any]:
        adapter = modelManager.get_adapter("asr")
        if not adapter or not adapter.is_loaded():
            raise RuntimeError("ASR model not loaded")

        return adapter.run_inference(audio_data, sample_rate=sample_rate, language=language)

asrService = ASRService()
