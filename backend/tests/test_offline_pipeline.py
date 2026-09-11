import unittest
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.services.model_manager import modelManager
from backend.services.asr_service import asrService
from backend.services.translation_service import translationService
from backend.services.tts_service import ttsService
from backend.services.benchmark_service import benchmarkService
from backend.services.audio_service import audioService
from backend.adapters.base import TTSStyleParams

class TestOfflineTranslationPipeline(unittest.TestCase):

    def setUp(self):
        # Ensure default models are loaded for testing
        modelManager.load_model("asr", "indic-conformer-all-indic-int8")
        modelManager.load_model("translation", "indictrans2-indic-en-dist-200m")
        modelManager.load_model("tts", "indic-multilingual-tts")

    def test_01_end_to_end_pipeline_benchmark(self):
        """Verify full offline pipeline runs and computes RTF accurately."""
        # Generate 2-second test audio
        test_wav = audioService.create_sample_wav(duration_sec=2.0)
        self.assertTrue(len(test_wav) > 1000)

        # Execute pipeline
        result = benchmarkService.run_benchmark(
            audio_bytes=test_wav,
            src_lang="hi",
            tgt_lang="en"
        )

        self.assertIn("realtime_factor", result)
        self.assertIn("total_latency_ms", result)
        self.assertIn("models", result)
        self.assertIn("results", result)
        
        # Verify transcript and translation are not empty
        self.assertTrue(len(result["results"]["source_transcript"]) > 0)
        self.assertTrue(len(result["results"]["translated_text"]) > 0)

        # Verify RTF formula: processing time / audio duration
        expected_rtf = round(result["processing_time_sec"] / result["audio_duration_sec"], 3)
        self.assertAlmostEqual(result["realtime_factor"], expected_rtf, places=2)

    def test_02_strict_asr_model_unloaded_rejection(self):
        """Verify that if ASR model is unloaded, system halts and throws 'ASR model not loaded'."""
        modelManager.unload_model("asr", "indic-conformer-all-indic-int8")
        test_wav = audioService.create_sample_wav(1.0)

        with self.assertRaises(RuntimeError) as ctx:
            asrService.transcribe(test_wav)

        self.assertIn("ASR model not loaded", str(ctx.exception))

        # Re-load for subsequent tests
        modelManager.load_model("asr", "indic-conformer-all-indic-int8")

    def test_03_strict_translation_model_unloaded_rejection(self):
        """Verify that if Translation model is unloaded, system halts and throws 'Translation model not loaded'."""
        modelManager.unload_model("translation", "indictrans2-indic-en-dist-200m")

        with self.assertRaises(RuntimeError) as ctx:
            translationService.translate("नमस्ते")

        self.assertIn("Translation model not loaded", str(ctx.exception))

        # Re-load
        modelManager.load_model("translation", "indictrans2-indic-en-dist-200m")

    def test_04_expressive_tts_parameters(self):
        """Verify TTS service supports and accepts expressive prosody parameters."""
        style_params = {
            "speed": 1.2,
            "pitch": 1.1,
            "energy": 0.9,
            "style": "conversational",
            "emotion": "happy",
            "expressiveness": 0.8
        }
        res = ttsService.generate("Where is the pharmacy?", language="en", style_params=style_params)
        self.assertIn("audio_bytes", res)
        self.assertIn("style_params_used", res)
        self.assertEqual(res["style_params_used"]["emotion"], "happy")
        self.assertEqual(res["style_params_used"]["speed"], 1.2)
        self.assertTrue(len(res["audio_bytes"]) > 500)

    def test_05_rtf_performance_classification(self):
        """Verify green/yellow/red RTF performance rating logic."""
        green = benchmarkService.compute_performance_rating(0.40)
        self.assertEqual(green["color"], "green")

        yellow = benchmarkService.compute_performance_rating(0.85)
        self.assertEqual(yellow["color"], "yellow")

        red = benchmarkService.compute_performance_rating(1.45)
        self.assertEqual(red["color"], "red")

if __name__ == "__main__":
    unittest.main()
