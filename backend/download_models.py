import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASR_DIR = MODELS_DIR / "asr" / "whisper-tiny-indic"
TTS_DIR = MODELS_DIR / "tts" / "piper-vits-en-lessac"
TRANS_DIR = MODELS_DIR / "translation" / "opus-mt-hi-en-int8"

for d in [ASR_DIR, TTS_DIR, TRANS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path, min_size_bytes: int = 1000):
    if dest_path.exists() and dest_path.stat().st_size >= min_size_bytes:
        print(f"[CACHE] {dest_path.name} already exists ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping.")
        return True

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out_f:
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024 # 1 MB chunks
            
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  -> {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end='\r')
        
        dur = time.time() - t0
        print(f"\n[DONE] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB in {dur:.1f}s)")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_asr_whisper():
    print("\n" + "="*60)
    print("1. DOWNLOADING ASR: Whisper Tiny Indic/Hindi (INT8 ONNX)")
    print("="*60)
    base_url = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main/"
    files = [
        ("tiny-encoder.int8.onnx", 12 * 1024 * 1024),
        ("tiny-decoder.int8.onnx", 80 * 1024 * 1024),
        ("tiny-tokens.txt", 500 * 1024)
    ]
    for filename, min_bytes in files:
        ok = download_file(base_url + filename, ASR_DIR / filename, min_bytes)
        if not ok:
            return False
    return True

def download_tts_piper():
    print("\n" + "="*60)
    print("2. DOWNLOADING TTS: Piper English Lessac (ONNX)")
    print("="*60)
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/"
    files = [
        ("en_US-lessac-low.onnx", 55 * 1024 * 1024),
        ("en_US-lessac-low.onnx.json", 1000)
    ]
    for filename, min_bytes in files:
        ok = download_file(base_url + filename, TTS_DIR / filename, min_bytes)
        if not ok:
            return False
    return True

def download_translation_opus():
    print("\n" + "="*60)
    print("3. DOWNLOADING TRANSLATION: Opus-MT Hindi->English (Quantized ONNX)")
    print("="*60)
    base_url = "https://huggingface.co/Xenova/opus-mt-hi-en/resolve/main/"
    files = [
        ("onnx/encoder_model_quantized.onnx", 45 * 1024 * 1024, "encoder_model_quantized.onnx"),
        ("onnx/decoder_model_merged_quantized.onnx", 50 * 1024 * 1024, "decoder_model_merged_quantized.onnx"),
        ("source.spm", 500 * 1024, "source.spm"),
        ("target.spm", 500 * 1024, "target.spm"),
        ("vocab.json", 100 * 1024, "vocab.json"),
        ("config.json", 500, "config.json")
    ]
    for item in files:
        remote_file = item[0]
        min_bytes = item[1]
        local_name = item[2]
        ok = download_file(base_url + remote_file, TRANS_DIR / local_name, min_bytes)
        if not ok:
            return False
    return True

if __name__ == "__main__":
    asr_ok = download_asr_whisper()
    tts_ok = download_tts_piper()
    trans_ok = download_translation_opus()

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"ASR Model: {'INSTALLED' if asr_ok else 'FAILED'}")
    print(f"TTS Model: {'INSTALLED' if tts_ok else 'FAILED'}")
    print(f"Translation Model: {'INSTALLED' if trans_ok else 'FAILED'}")
