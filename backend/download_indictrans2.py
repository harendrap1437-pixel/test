import os
import sys
import time
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent.parent / "models" / "translation" / "indictrans2-indic-en-dist-200m"
DEST_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/hari31416/indictrans2-indic-en-dist-200M-ONNX-int8/resolve/main/"

FILES = [
    "encoder_model.onnx",
    "encoder_model.onnx.data",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
    "decoder_shared.onnx.data",
    "config.json",
    "generation_config.json",
    "dict.SRC.json",
    "dict.TGT.json",
    "model.SRC",
    "model.TGT",
    "tokenizer_meta.json",
    "tokenizer_src.json",
    "tokenizer_tgt.json",
    "tokenization_indictrans.py",
    "translate.py"
]

def download_file(filename):
    dest = DEST_DIR / filename
    url = BASE_URL + filename
    print(f"[DOWNLOADING] Checking {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
            if dest.exists() and total_size > 0 and dest.stat().st_size == total_size:
                print(f"[CACHE] {filename} already exists and matches expected size ({dest.stat().st_size / (1024*1024):.2f} MB).")
                return True

            print(f"[DOWNLOADING] {filename} ({total_size/(1024*1024):.2f} MB)...")
            with open(dest, 'wb') as out_f:
                downloaded = 0
                while chunk := resp.read(1024*1024):
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"  -> {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%)", end='\r')
        dur = time.time() - t0
        print(f"\n[DONE] {filename} ({dest.stat().st_size / (1024*1024):.2f} MB in {dur:.1f}s)")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed {filename}: {e}")
        if dest.exists():
            dest.unlink()
        return False

if __name__ == "__main__":
    print("="*60)
    print("DOWNLOADING OFFICIAL INDICTRANS2 DISTILLED 200M INT8 ONNX")
    print("Source: https://huggingface.co/hari31416/indictrans2-indic-en-dist-200M-ONNX-int8")
    print("="*60)
    success = True
    for f in FILES:
        if not download_file(f):
            success = False
            break
    print(f"\nIndicTrans2 Download Result: {'SUCCESS' if success else 'FAILED'}")
