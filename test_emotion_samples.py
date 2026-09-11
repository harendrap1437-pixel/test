from pathlib import Path
from backend.adapters.emotion2vec import Emotion2VecPlusAdapter

adapter = Emotion2VecPlusAdapter(model_dir="models/emotion/emotion2vec-plus-base")
adapter.load()
print(f"Loaded {adapter.metadata.name} in {adapter.metadata.load_time_ms} ms\n")

samples_dir = Path("backend/data/emotion_samples")
print(f"{'Sample File':<26} {'Detected Emotion':<18} {'Confidence':<12} {'Latency (ms)':<12}")
print("-" * 70)
for wav_file in sorted(samples_dir.glob("*.wav")):
    audio_bytes = wav_file.read_bytes()
    res = adapter.run_inference(audio_bytes)
    print(f"{wav_file.name:<26} {res['emotion'].upper():<18} {res['confidence']*100:>5.1f}%      {res['latency_ms']:>6.1f} ms")
    top_scores = [f"{k}: {v*100:.1f}%" for k, v in sorted(res['scores'].items(), key=lambda x: x[1], reverse=True)[:4]]
    print("   -> " + " | ".join(top_scores))
