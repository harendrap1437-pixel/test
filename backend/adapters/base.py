from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class TTSStyleParams:
    speed: float = 1.0
    pitch: float = 1.0
    energy: float = 1.0
    style: str = "neutral"
    emotion: str = "neutral"
    expressiveness: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ModelMetadata:
    id: str
    name: str
    category: str  # 'asr', 'translation', 'tts'
    version: str
    size_mb: float
    format: str    # 'ONNX', 'ONNX-INT8', 'CTranslate2', 'TorchScript'
    supported_languages: list
    ram_estimate_mb: float
    is_loaded: bool = False
    load_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    status: str = "Not Loaded"
    filepath: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseModelAdapter(ABC):
    """
    Standard interface for all local AI runtime model adapters.
    Guarantees that runtimes (Sherpa-ONNX, ONNXRuntime, CTranslate2, etc.)
    can be swapped without modifying the application or UI layers.
    """
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self._is_loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Loads model weights into local memory from storage."""
        pass

    @abstractmethod
    def unload(self) -> bool:
        """Unloads model weights and frees memory."""
        pass

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_metadata(self) -> ModelMetadata:
        self.metadata.is_loaded = self._is_loaded
        return self.metadata

    @abstractmethod
    def run_inference(self, *args, **kwargs) -> Any:
        """Executes offline inference using loaded model weights."""
        pass

class ASRModelAdapter(BaseModelAdapter):
    """
    Adapter for Automatic Speech Recognition (IndicConformer / Sherpa-ONNX / Whisper).
    """
    @abstractmethod
    def run_inference(self, audio_data: bytes, sample_rate: int = 16000, language: str = "hi") -> Dict[str, Any]:
        """
        Input: Raw PCM or WAV audio bytes.
        Output: { "transcript": str, "language": str, "latency_ms": float }
        """
        pass

class TranslationModelAdapter(BaseModelAdapter):
    """
    Adapter for Local Machine Translation (IndicTrans2 CTranslate2 / ONNX).
    """
    @abstractmethod
    def run_inference(self, text: str, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
        """
        Input: Source text, source Indic language code, target language code.
        Output: { "translation": str, "src_lang": str, "tgt_lang": str, "latency_ms": float }
        """
        pass

class TTSModelAdapter(BaseModelAdapter):
    """
    Adapter for Local Text-to-Speech (Piper / VITS / Indic Parler ONNX).
    Includes expressive prosody parameters for future emotion adaptation.
    """
    @abstractmethod
    def run_inference(self, text: str, language: str, style_params: Optional[TTSStyleParams] = None) -> Dict[str, Any]:
        """
        Input: Text, language, style_params (speed, pitch, energy, style, emotion, expressiveness).
        Output: { "audio_bytes": bytes, "sample_rate": int, "format": str, "latency_ms": float }
        """
        pass

class EmotionModelAdapter(BaseModelAdapter):
    """
    Adapter for Local Speech Emotion Recognition (emotion2vec+ / ONNX).
    """
    @abstractmethod
    def run_inference(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Input: Raw PCM or WAV audio bytes.
        Output: {
            "emotion": str,
            "confidence": float,
            "scores": Dict[str, float],
            "latency_ms": float,
            "model_id": str,
            "status": str
        }
        """
        pass

    @abstractmethod
    def get_supported_emotions(self) -> list:
        pass

