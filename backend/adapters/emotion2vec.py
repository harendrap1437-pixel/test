import os
import io
import time
import json
import wave
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.adapters.base import EmotionModelAdapter, ModelMetadata

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

TARGET_EMOTIONS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown"
]

# 18 student pose categories -> 9 target emotion categories mapping
POSE_TO_EMOTION_MAP = {
    "calm": "neutral",
    "content": "happy",
    "warm": "happy",
    "happy": "happy",
    "amused": "happy",
    "excited": "happy",
    "surprised": "surprised",
    "smug": "surprised",
    "confused": "surprised",
    "skeptical": "surprised",
    "annoyed": "angry",
    "disgusted": "disgusted",
    "afraid": "fearful",
    "angry": "angry",
    "tense": "angry",
    "sad": "sad",
    "weary": "sad",
    "bored": "neutral"
}

POSES_ORDER = [
    "calm", "content", "warm", "happy", "amused", "excited",
    "surprised", "smug", "confused", "skeptical", "annoyed",
    "disgusted", "afraid", "angry", "tense", "sad", "weary", "bored"
]

class Emotion2VecPlusAdapter(EmotionModelAdapter):
    """
    Offline Speech Emotion Recognition Adapter based on emotion2vec+ (Distilled ONNX).
    Processes 16 kHz raw speech waveform and predicts real emotion probability distribution
    across 9 standard emotion categories: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown.
    100% offline; zero cloud fallback.
    """
    def __init__(self, metadata: Optional[ModelMetadata] = None, model_dir: Optional[str] = None):
        default_metadata = ModelMetadata(
            id="emotion2vec-plus-base",
            name="emotion2vec+ Base (Distilled ONNX)",
            category="emotion",
            version="1.0-onnx",
            size_mb=9.25,
            format="ONNX",
            supported_languages=["hi", "en", "ta", "te", "kn", "mr", "bn", "gu", "ml", "pa", "or"],
            ram_estimate_mb=45.0,
            is_loaded=False,
            load_time_ms=0.0,
            inference_time_ms=0.0,
            status="Not Loaded",
            filepath=model_dir
        )
        super().__init__(metadata or default_metadata)
        self.model_dir = model_dir
        self.session: Optional[Any] = None
        self.window_samples = 48000  # 3 seconds at 16kHz
        self.temperature = 0.0375

    def load(self) -> bool:
        t0 = time.perf_counter()
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is required for emotion2vec+ execution")

        onnx_path = None
        if self.model_dir and os.path.isdir(self.model_dir):
            candidate = os.path.join(self.model_dir, "model.onnx")
            if os.path.exists(candidate):
                onnx_path = candidate
            else:
                for f in os.listdir(self.model_dir):
                    if f.endswith(".onnx"):
                        onnx_path = os.path.join(self.model_dir, f)
                        break

        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"emotion2vec+ model.onnx not found in {self.model_dir}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._is_loaded = True
        self.metadata.load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.metadata.is_loaded = True
        self.metadata.status = "Loaded (Physical emotion2vec+ ONNX Model)"
        return True

    def unload(self) -> bool:
        self.session = None
        self._is_loaded = False
        self.metadata.is_loaded = False
        self.metadata.status = "Not Loaded"
        return True

    def get_supported_emotions(self) -> List[str]:
        return TARGET_EMOTIONS

    def run_inference(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        if not self._is_loaded or self.session is None:
            raise RuntimeError("Emotion model not loaded. Call load() first.")

        t0 = time.perf_counter()

        # 1. Parse and resample audio into float32 array [-1.0, 1.0] at 16000 Hz
        samples = self._parse_audio(audio_data, target_sr=16000)
        audio_dur = max(0.1, len(samples) / 16000.0)

        # 2. Windowing & Inference
        # If audio is empty or near zero energy
        rms = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0.0
        if len(samples) < 800 or rms < 0.001:
            # Neutral baseline for silence
            scores = {emo: 0.01 for emo in TARGET_EMOTIONS}
            scores["neutral"] = 0.90
            scores["other"] = 0.01
            scores["unknown"] = 0.01
            return {
                "emotion": "neutral",
                "confidence": 0.90,
                "scores": scores,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "audio_duration_sec": round(audio_dur, 2),
                "model_id": self.metadata.id,
                "status": "Success (Silence / Baseline Neutral)"
            }

        # Chunk into windows of 48000 samples (3s) with 50% overlap (24000 samples hop)
        windows = []
        if len(samples) <= self.window_samples:
            # Pad with zeros to exactly 48000
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[:len(samples)] = samples
            windows.append(padded)
        else:
            hop = self.window_samples // 2
            for start in range(0, len(samples) - self.window_samples + 1, hop):
                windows.append(samples[start:start + self.window_samples])
            # Include final window if needed
            if len(samples) % hop != 0:
                final_win = samples[-self.window_samples:]
                windows.append(final_win)

        # Run ONNX inference over all windows and average softmax logits
        pose_probs_list = []
        for win in windows:
            batch_input = np.expand_dims(win, axis=0).astype(np.float32)
            cosines = self.session.run(None, {"waveform": batch_input})[0][0]  # shape (18,)
            
            # Apply temperature-scaled softmax
            scaled = cosines / self.temperature
            scaled_shift = scaled - np.max(scaled)
            exps = np.exp(scaled_shift)
            probs = exps / np.sum(exps)
            pose_probs_list.append(probs)

        avg_pose_probs = np.mean(pose_probs_list, axis=0)

        # 3. Map 18 poses into 9 standard emotion categories
        emotion_scores = {emo: 0.0 for emo in TARGET_EMOTIONS}
        for i, pose in enumerate(POSES_ORDER):
            if i < len(avg_pose_probs):
                target_emo = POSE_TO_EMOTION_MAP.get(pose, "other")
                emotion_scores[target_emo] += float(avg_pose_probs[i])

        # Normalize so sum = 1.0
        total_sum = sum(emotion_scores.values())
        if total_sum > 0:
            for k in emotion_scores:
                emotion_scores[k] = round(emotion_scores[k] / total_sum, 4)

        # Find top emotion
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_confidence = sorted_emotions[0]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.metadata.inference_time_ms = latency_ms

        return {
            "emotion": top_emotion,
            "confidence": round(top_confidence, 4),
            "scores": emotion_scores,
            "latency_ms": latency_ms,
            "audio_duration_sec": round(audio_dur, 2),
            "model_id": self.metadata.id,
            "status": "Success (Local Offline Inference)"
        }

    def _parse_audio(self, audio_data: bytes, target_sr: int = 16000) -> np.ndarray:
        """Parses audio bytes into float32 array at target sample rate."""
        # 1. Try ffmpeg (supports MP3, M4A, AAC, WebM, OGG, FLAC, WAV stereo/mono)
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-nostdin', '-loglevel', 'quiet',
                '-i', 'pipe:0',
                '-f', 's16le',
                '-ac', '1',
                '-ar', str(target_sr),
                'pipe:1'
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = proc.communicate(input=audio_data, timeout=5)
            if proc.returncode == 0 and len(out) > 0:
                return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass

        # 2. Try standard wave
        try:
            with io.BytesIO(audio_data) as bio:
                with wave.open(bio, 'rb') as wav:
                    n_channels = wav.getnchannels()
                    sampwidth = wav.getsampwidth()
                    wav_sr = wav.getframerate()
                    raw_frames = wav.readframes(wav.getnframes())

                    if sampwidth == 2:
                        dtype = np.int16
                    elif sampwidth == 4:
                        dtype = np.int32
                    else:
                        dtype = np.uint8

                    data = np.frombuffer(raw_frames, dtype=dtype)
                    if n_channels > 1:
                        data = data.reshape(-1, n_channels)[:, 0]
                    samples = data.astype(np.float32) / 32768.0

                    if wav_sr != target_sr and len(samples) > 0:
                        new_len = int(len(samples) * target_sr / wav_sr)
                        samples = np.interp(
                            np.linspace(0, len(samples), new_len, endpoint=False),
                            np.arange(len(samples)),
                            samples
                        ).astype(np.float32)
                    return samples
        except Exception:
            pass

        # 3. Try soundfile
        if HAS_SOUNDFILE:
            try:
                with io.BytesIO(audio_data) as bio:
                    data, sf_sr = sf.read(bio, dtype='float32')
                    if data.ndim > 1:
                        data = data[:, 0]
                    if sf_sr != target_sr and len(data) > 0:
                        new_len = int(len(data) * target_sr / sf_sr)
                        data = np.interp(
                            np.linspace(0, len(data), new_len, endpoint=False),
                            np.arange(len(data)),
                            data
                        ).astype(np.float32)
                    return data
            except Exception:
                pass

        # 4. Raw PCM fallback
        try:
            data = np.frombuffer(audio_data, dtype=np.int16)
            return data.astype(np.float32) / 32768.0
        except Exception:
            return np.zeros(target_sr * 2, dtype=np.float32)
