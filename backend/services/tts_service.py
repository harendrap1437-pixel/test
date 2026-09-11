from typing import Dict, Any, Optional
from backend.services.model_manager import modelManager
from backend.adapters.base import TTSStyleParams

class TTSService:
    """
    Offline Text-to-Speech Service.
    Supports expressive speech synthesis parameters:
    speed, pitch, energy, style, emotion, expressiveness.
    Zero cloud TTS fallback.
    """
    def generate(
        self,
        text: str,
        language: str = "en",
        style_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        adapter = modelManager.get_adapter("tts")
        if not adapter or not adapter.is_loaded():
            raise RuntimeError("TTS model not loaded")

        params_obj = TTSStyleParams()
        if style_params:
            if "speed" in style_params:
                params_obj.speed = float(style_params["speed"])
            if "pitch" in style_params:
                params_obj.pitch = float(style_params["pitch"])
            if "energy" in style_params:
                params_obj.energy = float(style_params["energy"])
            if "style" in style_params:
                params_obj.style = str(style_params["style"])
            if "emotion" in style_params:
                params_obj.emotion = str(style_params["emotion"])
            if "expressiveness" in style_params:
                params_obj.expressiveness = float(style_params["expressiveness"])

        return adapter.run_inference(text, language=language, style_params=params_obj)

ttsService = TTSService()
