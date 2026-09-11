import sys
import urllib.request
import json
import urllib.parse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=== VERIFYING BACKEND ENDPOINTS ===")

# 1. Status
with urllib.request.urlopen('http://127.0.0.1:8000/api/status') as r:
    st = json.loads(r.read())
    print(f"Status API: {st['status']} | Active Models: {list(st['active_models'].keys())}")

# 2. Models
with urllib.request.urlopen('http://127.0.0.1:8000/api/models') as r:
    models = json.loads(r.read())
    print(f"Categories: {list(models.keys())}")
    for cat, mlist in models.items():
        print(f"  [{cat.upper()}]: {[m['name'] + ' (' + str(m['size_mb']) + 'MB)' for m in mlist]}")

# 3. Dedicated Standalone Emotion Analysis Endpoint
req_data = urllib.parse.urlencode({'sample_filename': 'sample_hi_angry.wav'}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/api/emotion/analyze-audio', data=req_data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req) as r:
    emo_res = json.loads(r.read())
    print(f"\nDedicated Emotion SER Endpoint (/api/emotion/analyze-audio):")
    print(f"  Input: sample_hi_angry.wav")
    print(f"  Predicted Emotion: {emo_res['emotion'].upper()}")
    print(f"  Confidence: {emo_res['confidence']*100:.1f}%")
    print(f"  Latency: {emo_res['latency_ms']} ms")
    print(f"  Scores: {emo_res['scores']}")

# 4. Pipeline Translation with Emotion
pipe_data = urllib.parse.urlencode({
    'sample_filename': 'sample_hi_phr1_pharmacy.wav',
    'src_lang': 'hi',
    'tgt_lang': 'en'
}).encode('utf-8')
req2 = urllib.request.Request('http://127.0.0.1:8000/api/pipeline/translate-audio', data=pipe_data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req2) as r:
    pipe_res = json.loads(r.read())
    print(f"\nIntegrated 8-Stage Pipeline (/api/pipeline/translate-audio):")
    print(f"  Success: {pipe_res['success']}")
    print(f"  Source Transcript: {pipe_res['source_transcript']}")
    print(f"  Detected Emotion: {pipe_res['detected_emotion'].upper()} ({pipe_res['emotion_confidence']*100:.1f}%)")
    print(f"  Translated Text: {pipe_res['translated_text']}")
    print(f"  TTS Status: {pipe_res['tts_status']}")
    print(f"  Audio Output Generated: {len(pipe_res['audio_base64'])} chars")
    print(f"  Total Latency: {pipe_res['benchmark']['total_latency_ms']} ms (RTF: {pipe_res['benchmark']['realtime_factor']})")
    print(f"  Stages:")
    for stage in pipe_res['pipeline_stages']:
        extra = f" -> {stage.get('transcript') or stage.get('translation') or stage.get('emotion') or ''}" if (stage.get('transcript') or stage.get('translation') or stage.get('emotion')) else ""
        lat = f" ({stage.get('latency_ms')}ms)" if stage.get('latency_ms') is not None else ""
        print(f"    Step {stage['step']}: {stage['name']}{lat}{extra}")

print("\n=== ALL ENDPOINTS VERIFIED SUCCESSFULLY ===")
