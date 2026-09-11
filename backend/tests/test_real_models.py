import os
import sys
import time
import json
import wave
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

def test_whisper_asr():
    print("="*60)
    print("TEST 1: Real Local Whisper Tiny Indic ASR")
    print("="*60)
    import sherpa_onnx
    
    model_dir = BASE_DIR / "models" / "asr" / "whisper-tiny-indic"
    encoder = str(model_dir / "tiny-encoder.int8.onnx")
    decoder = str(model_dir / "tiny-decoder.int8.onnx")
    tokens = str(model_dir / "tiny-tokens.txt")

    t0 = time.perf_counter()
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        language="hi",
        task="transcribe",
        num_threads=2
    )
    load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"Loaded successfully in {load_time_ms} ms")

    # Ingest sample wave
    sample_wav = BASE_DIR / "models" / "sample_audio" / "sample_hi_phr1_pharmacy.wav"
    with wave.open(str(sample_wav), "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    t_inf = time.perf_counter()
    stream = rec.create_stream()
    stream.accept_waveform(16000, samples)
    rec.decode_stream(stream)
    text = stream.result.text
    inf_time_ms = round((time.perf_counter() - t_inf) * 1000, 2)

    print(f"Inference Time: {inf_time_ms} ms")
    print(f"Recognized Transcript: '{text}'")
    return {
        "model": "Whisper Tiny Indic (INT8 ONNX)",
        "load_time_ms": load_time_ms,
        "inference_time_ms": inf_time_ms,
        "transcript": text
    }

def test_piper_tts():
    print("\n" + "="*60)
    print("TEST 2: Real Local Piper English TTS (ONNX)")
    print("="*60)
    import sherpa_onnx
    
    tts_dir = BASE_DIR / "models" / "tts" / "piper-vits-en-lessac"
    model = str(tts_dir / "en_US-lessac-low.onnx")
    tokens = str(tts_dir / "en_US-lessac-low.onnx.json")

    t0 = time.perf_counter()
    # Sherpa-ONNX VITS Piper config
    vits_cfg = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=model,
        tokens=tokens,
        data_dir=""
    )
    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_cfg,
        num_threads=2,
        debug=False
    )
    tts_cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
    tts = sherpa_onnx.OfflineTts(tts_cfg)
    load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"Piper TTS loaded in {load_time_ms} ms")

    test_sentence = "Where is the pharmacy? I need medicine."
    t_inf = time.perf_counter()
    audio = tts.generate(test_sentence, sid=0, speed=1.0)
    inf_time_ms = round((time.perf_counter() - t_inf) * 1000, 2)

    samples_count = len(audio.samples)
    audio_dur = round(samples_count / audio.sample_rate, 2)
    print(f"Inference Time: {inf_time_ms} ms")
    print(f"Generated Audio: {audio_dur}s ({samples_count} samples at {audio.sample_rate}Hz)")
    
    return {
        "model": "Piper English Lessac (ONNX)",
        "load_time_ms": load_time_ms,
        "inference_time_ms": inf_time_ms,
        "audio_duration_sec": audio_dur,
        "sample_rate": audio.sample_rate
    }

def test_opus_translation():
    print("\n" + "="*60)
    print("TEST 3: Real Local MarianMT/Opus Hindi->English (Quantized ONNX)")
    print("="*60)
    import onnxruntime as ort
    import sentencepiece as spm

    trans_dir = BASE_DIR / "models" / "translation" / "opus-mt-hi-en-int8"
    enc_path = str(trans_dir / "encoder_model_quantized.onnx")
    dec_path = str(trans_dir / "decoder_model_merged_quantized.onnx")
    src_spm_path = str(trans_dir / "source.spm")
    tgt_spm_path = str(trans_dir / "target.spm")

    t0 = time.perf_counter()
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    enc_sess = ort.InferenceSession(enc_path, sess_opts, providers=['CPUExecutionProvider'])
    dec_sess = ort.InferenceSession(dec_path, sess_opts, providers=['CPUExecutionProvider'])
    
    sp_src = spm.SentencePieceProcessor(model_file=src_spm_path)
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_spm_path)
    load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"ONNX Translation models loaded in {load_time_ms} ms")

    hindi_text = "भाई फार्मेसी किधर है? मुझे दवा चाहिए।"
    t_inf = time.perf_counter()
    
    # 1. Encode with source SentencePiece
    tokens = sp_src.encode(hindi_text, out_type=int)
    # Add EOS token (typically id 0 in Marian)
    input_ids = np.array([tokens + [0]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    # 2. Run encoder
    enc_inputs = {
        enc_sess.get_inputs()[0].name: input_ids,
        enc_sess.get_inputs()[1].name: attention_mask
    }
    enc_outputs = enc_sess.run(None, enc_inputs)
    last_hidden_state = enc_outputs[0]

    # 3. Greedy Decode with merged decoder
    dec_input_ids = np.array([[61877]], dtype=np.int64) # decoder start token (or initial token)
    generated_tokens = []
    
    # Greedy step
    for _ in range(25):
        # Inspect merged decoder inputs
        dec_inputs = {}
        for inp in dec_sess.get_inputs():
            if inp.name == "input_ids":
                dec_inputs[inp.name] = dec_input_ids
            elif inp.name == "encoder_hidden_states":
                dec_inputs[inp.name] = last_hidden_state
            elif inp.name == "encoder_attention_mask":
                dec_inputs[inp.name] = attention_mask
            elif inp.name == "use_cache_branch":
                dec_inputs[inp.name] = np.array([False])
        
        logits = dec_sess.run(None, dec_inputs)[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == 0: # EOS
            break
        generated_tokens.append(next_token)
        dec_input_ids = np.concatenate([dec_input_ids, [[next_token]]], axis=1)

    translated_text = sp_tgt.decode(generated_tokens)
    inf_time_ms = round((time.perf_counter() - t_inf) * 1000, 2)
    print(f"Inference Time: {inf_time_ms} ms")
    print(f"Translated Text: '{translated_text}'")

    return {
        "model": "Opus-MT Hindi->English (Quantized ONNX)",
        "load_time_ms": load_time_ms,
        "inference_time_ms": inf_time_ms,
        "translated_text": translated_text
    }

if __name__ == "__main__":
    asr_res = test_whisper_asr()
    tts_res = test_piper_tts()
    try:
        tr_res = test_opus_translation()
    except Exception as e:
        print(f"Translation test note: {e}")
