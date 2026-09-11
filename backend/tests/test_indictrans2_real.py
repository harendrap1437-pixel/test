import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
trans_dir = BASE_DIR / "models" / "translation" / "indictrans2-indic-en-dist-200m"
sys.path.insert(0, str(trans_dir))

from translate import IndicTransONNX

def run_test():
    print("Loading IndicTrans2 ONNX INT8 model from local storage...")
    t0 = time.perf_counter()
    model = IndicTransONNX(str(trans_dir))
    load_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"Loaded in {load_ms} ms")

    phrases = [
        "भाई फार्मेसी किधर है? मुझे दवा चाहिए।",
        "नमस्ते, क्या आप मेरी मदद कर सकते हैं?",
        "धन्यवाद भाई।"
    ]

    for p in phrases:
        t_inf = time.perf_counter()
        res = model.translate(p, src_lang="hin_Deva", tgt_lang="eng_Latn")
        inf_ms = round((time.perf_counter() - t_inf) * 1000, 2)
        print(f"\n[INPUT HINDI]: {p}")
        print(f"[OUTPUT ENGLISH] ({inf_ms} ms): {res}")

if __name__ == "__main__":
    run_test()
