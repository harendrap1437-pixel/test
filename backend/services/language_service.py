from typing import Dict, List, Optional

class LanguageService:
    """
    Manages supported Indic and global languages, ISO codes, and IndicTrans2 tags.
    """
    LANGUAGES = {
        "hi": {
            "name": "Hindi",
            "native": "हिन्दी",
            "indictrans2_tag": "hin_Deva",
            "supported_in_poc": True
        },
        "en": {
            "name": "English",
            "native": "English",
            "indictrans2_tag": "eng_Latn",
            "supported_in_poc": True
        },
        "ta": {
            "name": "Tamil",
            "native": "தமிழ்",
            "indictrans2_tag": "tam_Taml",
            "supported_in_poc": True
        },
        "te": {
            "name": "Telugu",
            "native": "తెలుగు",
            "indictrans2_tag": "tel_Telu",
            "supported_in_poc": True
        },
        "kn": {
            "name": "Kannada",
            "native": "ಕನ್ನಡ",
            "indictrans2_tag": "kan_Knda",
            "supported_in_poc": True
        },
        "bn": {
            "name": "Bengali",
            "native": "বাংলা",
            "indictrans2_tag": "ben_Beng",
            "supported_in_poc": True
        },
        "mr": {
            "name": "Marathi",
            "native": "मराठी",
            "indictrans2_tag": "mar_Deva",
            "supported_in_poc": True
        }
    }

    def get_supported_languages(self) -> List[Dict]:
        return [
            {
                "code": code,
                "name": data["name"],
                "native": data["native"],
                "tag": data["indictrans2_tag"]
            }
            for code, data in self.LANGUAGES.items()
        ]

    def get_tag(self, code: str) -> str:
        return self.LANGUAGES.get(code, {}).get("indictrans2_tag", "eng_Latn")

    def is_valid_pair(self, src: str, tgt: str) -> bool:
        return src in self.LANGUAGES and tgt in self.LANGUAGES and src != tgt

languageService = LanguageService()
