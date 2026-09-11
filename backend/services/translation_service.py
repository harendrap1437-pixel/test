from typing import Dict, Any, Optional
from backend.services.model_manager import modelManager

class TranslationService:
    """
    Offline Machine Translation Service.
    Delegates to the currently active TranslationModelAdapter.
    Zero cloud translation fallback.
    """
    def translate(self, text: str, src_lang: str = "hi", tgt_lang: str = "en") -> Dict[str, Any]:
        adapter = modelManager.get_adapter("translation")
        if not adapter or not adapter.is_loaded():
            raise RuntimeError("Translation model not loaded")

        return adapter.run_inference(text, src_lang=src_lang, tgt_lang=tgt_lang)

translationService = TranslationService()
