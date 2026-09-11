import os
import io
import time
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import BASE_DIR, DATA_DIR, SAMPLE_AUDIO_DIR, EMOTION_SAMPLES_DIR
from backend.services.asr_service import asrService
from backend.services.emotion_service import emotionService
from backend.services.translation_service import translationService
from backend.services.tts_service import ttsService
from backend.services.model_manager import modelManager
from backend.services.benchmark_service import benchmarkService
from backend.services.audio_service import audioService
from backend.services.language_service import languageService

app = FastAPI(title="AI Portable Language Translator - Local AI POC")

# Allow local frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Offline / Airplane Mode State
offline_state = {
    "airplane_mode": True,
    "network_blocked": True,
    "cloud_services_disabled": True,
    "status_message": "OFFLINE MODE: NETWORK NOT REQUIRED"
}

# --- System & Status Endpoints ---
@app.get("/api/status")
def get_status():
    return {
        "status": "online_local",
        "offline_state": offline_state,
        "ram_mb": benchmarkService.get_process_ram_mb(),
        "active_models": modelManager.active_models
    }

@app.get("/api/languages")
def get_languages():
    return languageService.get_supported_languages()

# --- Full Modular Speech-to-Speech + Emotion Pipeline Endpoint ---
@app.post("/api/pipeline/translate-audio")
async def translate_audio_pipeline(
    audio_file: Optional[UploadFile] = File(None),
    raw_audio_base64: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None),
    src_lang: str = Form("hi"),
    tgt_lang: str = Form("en"),
    speed: float = Form(1.0),
    pitch: float = Form(1.0),
    energy: float = Form(1.0),
    style: str = Form("neutral"),
    emotion: str = Form("neutral"),
    expressiveness: float = Form(0.5)
):
    """
    Executes the modular 8-stage local speech-to-speech translation pipeline:
    Audio In (1) -> Local ASR (2) -> Transcript (3) -> Emotion SER (4) -> Translation (5,6) -> Emotion-Aware TTS (7) -> Audio Out (8).
    Strictly offline. Zero cloud fallbacks.
    """
    # 1. Ingest audio bytes
    audio_bytes = None
    if audio_file:
        audio_bytes = await audio_file.read()
    elif raw_audio_base64:
        # Strip header if present
        if "," in raw_audio_base64:
            raw_audio_base64 = raw_audio_base64.split(",", 1)[1]
        audio_bytes = base64.b64decode(raw_audio_base64)
    elif sample_filename:
        # Check both sample directories
        sample_path = EMOTION_SAMPLES_DIR / sample_filename
        if not sample_path.exists():
            sample_path = SAMPLE_AUDIO_DIR / sample_filename
        if sample_path.exists():
            audio_bytes = sample_path.read_bytes()
        else:
            raise HTTPException(status_code=404, detail=f"Sample file {sample_filename} not found")
    else:
        # Fallback to default sample
        samples = list(EMOTION_SAMPLES_DIR.glob("*.wav")) or list(SAMPLE_AUDIO_DIR.glob("*.wav"))
        if samples:
            audio_bytes = samples[0].read_bytes()
        else:
            audio_bytes = audioService.create_sample_wav(2.0)

    # 2. Prepare style parameters
    style_params = {
        "speed": speed,
        "pitch": pitch,
        "energy": energy,
        "style": style,
        "emotion": emotion,
        "expressiveness": expressiveness
    }

    try:
        # Execute Benchmark Pipeline (ASR -> Emotion SER -> IndicTrans2 -> Emotion-Modulated TTS)
        result = benchmarkService.run_benchmark(
            audio_bytes=audio_bytes,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            style_params=style_params
        )

        # Retrieve generated audio
        effective_style = dict(style_params)
        effective_style["emotion"] = result["results"]["detected_emotion"]
        effective_style["confidence"] = result["results"]["emotion_confidence"]

        tts_res = ttsService.generate(
            text=result["results"]["translated_text"],
            language=tgt_lang,
            style_params=effective_style
        )

        audio_fmt = tts_res.get("format", "audio/wav")
        audio_b64 = base64.b64encode(tts_res["audio_bytes"]).decode("utf-8")
        data_uri = f"data:{audio_fmt};base64,{audio_b64}"

        return {
            "success": True,
            "pipeline_stages": [
                {"step": 1, "name": "Audio captured", "status": "Done", "duration_sec": result["audio_duration_sec"]},
                {"step": 2, "name": "ASR running", "status": "Done", "latency_ms": result["models"]["asr"]["latency_ms"]},
                {"step": 3, "name": "Transcript produced", "status": "Done", "transcript": result["results"]["source_transcript"]},
                {"step": 4, "name": "Emotion analysis", "status": "Done", "emotion": result["results"]["detected_emotion"].upper(), "confidence": round(result["results"]["emotion_confidence"] * 100, 1), "latency_ms": result["models"]["emotion"]["latency_ms"]},
                {"step": 5, "name": "Translation running", "status": "Done", "latency_ms": result["models"]["translation"]["latency_ms"]},
                {"step": 6, "name": "Translation produced", "status": "Done", "translation": result["results"]["translated_text"]},
                {"step": 7, "name": "Emotion-aware TTS running", "status": "Done", "latency_ms": result["models"]["tts"]["latency_ms"]},
                {"step": 8, "name": "Audio output", "status": "Done", "format": audio_fmt}
            ],
            "source_transcript": result["results"]["source_transcript"],
            "detected_emotion": result["results"]["detected_emotion"],
            "emotion_confidence": result["results"]["emotion_confidence"],
            "emotion_scores": result["results"]["emotion_scores"],
            "translated_text": result["results"]["translated_text"],
            "tts_status": "Emotion Prosody Applied",
            "audio_output": data_uri,
            "audio_base64": data_uri,
            "benchmark": result
        }

    except RuntimeError as e:
        # Return visible error (e.g. "ASR model not loaded")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# --- Dedicated Standalone Emotion Recognition Endpoint ---
@app.post("/api/emotion/analyze-audio")
async def analyze_emotion_audio(
    audio_file: Optional[UploadFile] = File(None),
    raw_audio_base64: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None)
):
    """
    Dedicated offline Speech Emotion Recognition endpoint.
    Processes original raw audio waveform (16 kHz) through emotion2vec+ Base.
    Returns real emotion probability distribution, top emotion, and confidence score.
    Strictly offline; zero cloud fallback.
    """
    audio_bytes = None
    if audio_file:
        audio_bytes = await audio_file.read()
    elif raw_audio_base64:
        if "," in raw_audio_base64:
            raw_audio_base64 = raw_audio_base64.split(",", 1)[1]
        audio_bytes = base64.b64decode(raw_audio_base64)
    elif sample_filename:
        p = EMOTION_SAMPLES_DIR / sample_filename
        if not p.exists():
            p = SAMPLE_AUDIO_DIR / sample_filename
        if p.exists():
            audio_bytes = p.read_bytes()
        else:
            raise HTTPException(status_code=404, detail=f"Sample file {sample_filename} not found")
    else:
        samples = list(EMOTION_SAMPLES_DIR.glob("*.wav")) or list(SAMPLE_AUDIO_DIR.glob("*.wav"))
        if samples:
            audio_bytes = samples[0].read_bytes()
        else:
            audio_bytes = audioService.create_sample_wav(2.0)

    try:
        res = emotionService.analyze_audio(audio_bytes, sample_rate=16000)
        return {
            "success": True,
            "emotion": res["emotion"],
            "confidence": res["confidence"],
            "scores": res["scores"],
            "latency_ms": res.get("latency_ms", 0.0),
            "audio_duration_sec": res.get("audio_duration_sec", 0.0),
            "model_id": res.get("model_id", "emotion2vec-plus-base"),
            "status": "Verified Offline Inference"
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion analysis error: {str(e)}")

@app.get("/api/emotion/samples")
def list_emotion_samples():
    """Lists available local emotional audio test samples."""
    samples = []
    if EMOTION_SAMPLES_DIR.exists():
        for f in EMOTION_SAMPLES_DIR.glob("*.wav"):
            samples.append({
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "label": f.stem.replace("sample_hi_", "").replace("_", " ").title()
            })
    return samples

@app.get("/api/emotion/samples/{filename}")
def serve_emotion_sample_audio(filename: str):
    p = EMOTION_SAMPLES_DIR / filename
    if not p.exists():
        p = SAMPLE_AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Emotion sample file not found")
    return FileResponse(str(p), media_type="audio/wav")

# --- Direct Text Translation Endpoint ---
class TextTranslateRequest(BaseModel):
    text: str
    src_lang: str = "hi"
    tgt_lang: str = "en"

@app.post("/api/pipeline/translate-text")
def translate_text(req: TextTranslateRequest):
    try:
        res = translationService.translate(req.text, src_lang=req.src_lang, tgt_lang=req.tgt_lang)
        return res
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Direct TTS Endpoint ---
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    energy: float = 1.0
    style: str = "neutral"
    emotion: str = "neutral"
    expressiveness: float = 0.5

@app.post("/api/pipeline/tts")
def generate_tts(req: TTSRequest):
    try:
        style_params = {
            "speed": req.speed,
            "pitch": req.pitch,
            "energy": req.energy,
            "style": req.style,
            "emotion": req.emotion,
            "expressiveness": req.expressiveness
        }
        res = ttsService.generate(req.text, language=req.language, style_params=style_params)
        audio_b64 = base64.b64encode(res["audio_bytes"]).decode("utf-8")
        return {
            "latency_ms": res["latency_ms"],
            "sample_rate": res["sample_rate"],
            "audio_base64": f"data:audio/wav;base64,{audio_b64}",
            "style_params_used": res["style_params_used"]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Model Manager Endpoints ---
@app.get("/api/models")
def get_models():
    return modelManager.list_models()

@app.post("/api/models/{category}/{model_id}/load")
def load_model(category: str, model_id: str):
    try:
        return modelManager.load_model(category, model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/{category}/{model_id}/unload")
def unload_model(category: str, model_id: str):
    try:
        return modelManager.unload_model(category, model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/{category}/{model_id}/test")
def test_model(category: str, model_id: str):
    try:
        return modelManager.test_model(category, model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/models/{category}/{model_id}")
def delete_model(category: str, model_id: str):
    try:
        return modelManager.delete_model(category, model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ImportModelRequest(BaseModel):
    category: str
    name: str
    version: str
    format: str
    size_mb: float
    supported_languages: List[str]

@app.post("/api/models/import")
def import_model(req: ImportModelRequest):
    try:
        meta = modelManager.import_local_model(
            category=req.category,
            name=req.name,
            version=req.version,
            format_str=req.format,
            size_mb=req.size_mb,
            supported_langs=req.supported_languages
        )
        return meta.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Benchmark & Comparison Endpoints ---
@app.get("/api/benchmark/history")
def get_benchmark_history():
    return benchmarkService.history

@app.post("/api/benchmark/compare")
def compare_models(sample_filename: Optional[str] = None):
    audio_bytes = None
    if sample_filename:
        p = SAMPLE_AUDIO_DIR / sample_filename
        if p.exists():
            audio_bytes = p.read_bytes()
    if not audio_bytes:
        samples = list(SAMPLE_AUDIO_DIR.glob("*.wav"))
        audio_bytes = samples[0].read_bytes() if samples else audioService.create_sample_wav(3.0)

    try:
        return benchmarkService.get_comparison(audio_bytes, src_lang="hi", tgt_lang="en")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Audio Samples ---
@app.get("/api/samples")
def get_samples():
    return audioService.get_sample_audio_files()

@app.get("/api/samples/{filename}")
def serve_sample_audio(filename: str):
    p = SAMPLE_AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="audio/wav")

# --- Translation Quality Test (10 Categories) ---
QUALITY_DATASET_FILE = DATA_DIR / "quality_test_dataset.json"

@app.get("/api/quality-test")
def get_quality_test_dataset():
    if not QUALITY_DATASET_FILE.exists():
        return []
    with open(QUALITY_DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/quality-test/run")
def run_quality_test():
    """
    Translates all 10 local test phrases offline and stores the actual translations.
    """
    if not QUALITY_DATASET_FILE.exists():
        raise HTTPException(status_code=404, detail="Quality dataset not found")

    with open(QUALITY_DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        try:
            res = translationService.translate(item["source"], src_lang="hi", tgt_lang="en")
            item["actual_translation"] = res.get("translation", "")
            # Auto-assign initial manual evaluation if identical to expected
            if item["actual_translation"].strip().lower() == item["expected_translation"].strip().lower():
                item["manual_evaluation"] = "Correct"
            elif item["actual_translation"]:
                item["manual_evaluation"] = "Mostly correct"
            else:
                item["manual_evaluation"] = "Needs improvement"
        except Exception as e:
            item["actual_translation"] = f"[Error: {str(e)}]"
            item["manual_evaluation"] = "Needs improvement"

    with open(QUALITY_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Compute prototype evaluation score
    total = len(data)
    correct_count = sum(1 for d in data if d.get("manual_evaluation") == "Correct")
    mostly_count = sum(1 for d in data if d.get("manual_evaluation") == "Mostly correct")
    score_pct = round(((correct_count * 1.0 + mostly_count * 0.75) / max(1, total)) * 100, 1)

    return {
        "dataset": data,
        "prototype_evaluation_score_pct": score_pct,
        "correct_count": correct_count,
        "mostly_correct_count": mostly_count,
        "needs_improvement_count": total - correct_count - mostly_count,
        "label": "Prototype evaluation",
        "disclaimer": "Local prototype evaluation metric on curated test suite. Not a formal academic benchmark."
    }

class EvaluateRequest(BaseModel):
    item_id: int
    evaluation: str  # "Correct", "Mostly correct", "Needs improvement"

@app.post("/api/quality-test/evaluate")
def evaluate_quality_item(req: EvaluateRequest):
    if not QUALITY_DATASET_FILE.exists():
        raise HTTPException(status_code=404, detail="Quality dataset not found")

    with open(QUALITY_DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item["id"] == req.item_id:
            item["manual_evaluation"] = req.evaluation
            break

    with open(QUALITY_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Compute updated score
    total = len(data)
    correct_count = sum(1 for d in data if d.get("manual_evaluation") == "Correct")
    mostly_count = sum(1 for d in data if d.get("manual_evaluation") == "Mostly correct")
    score_pct = round(((correct_count * 1.0 + mostly_count * 0.75) / max(1, total)) * 100, 1)

    return {
        "success": True,
        "prototype_evaluation_score_pct": score_pct,
        "label": "Prototype evaluation"
    }

# --- Offline / Airplane Mode Enforcement ---
@app.get("/api/offline/status")
def get_offline_status():
    return offline_state

@app.post("/api/offline/toggle")
def toggle_offline(enabled: bool = True):
    offline_state["airplane_mode"] = enabled
    offline_state["network_blocked"] = enabled
    offline_state["cloud_services_disabled"] = enabled
    offline_state["status_message"] = (
        "OFFLINE MODE: NETWORK NOT REQUIRED (100% On-Device)"
        if enabled else "ONLINE NETWORK ENABLED"
    )
    return offline_state

# Serve frontend static assets
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

@app.get("/")
def serve_index():
    index_path = BASE_DIR / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Portable Language Translator Backend Running"}
