import os
import sys
import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EMOTION_MODEL_DIR = BASE_DIR / "models" / "emotion" / "emotion2vec-plus-base"
EMOTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://huggingface.co/thomashallock/emotion2vec-web-distill/resolve/main/distill-student-v4.fused.onnx"
META_URL = "https://huggingface.co/thomashallock/emotion2vec-web-distill/raw/main/distill-student-v4.fused-meta.json"
TOKENS_URL = "https://huggingface.co/emotion2vec/emotion2vec_plus_base/raw/main/tokens.txt"

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[Exists] {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return
    print(f"[Downloading] {url} -> {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    print(f"[Done] {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")

def main():
    print("=== Downloading emotion2vec+ Local ONNX Model ===")
    model_onnx = EMOTION_MODEL_DIR / "model.onnx"
    meta_json = EMOTION_MODEL_DIR / "meta.json"
    tokens_txt = EMOTION_MODEL_DIR / "tokens.txt"

    download_file(MODEL_URL, model_onnx)
    download_file(META_URL, meta_json)
    download_file(TOKENS_URL, tokens_txt)

    # Verify ONNX model can be loaded by onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_onnx), providers=["CPUExecutionProvider"])
        inputs = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        print(f"[Verification Success] ONNX model loaded! Inputs: {inputs}, Outputs: {outputs}")
    except Exception as e:
        print(f"[Verification Failed] {e}")

if __name__ == "__main__":
    main()
