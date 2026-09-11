# AI Portable Language Translator (Local AI POC)

> **100% Offline Speech-to-Speech Indic AI Translation Pipeline**  
> *Zero Browser SpeechRecognition • Zero Google Translate • Zero Cloud STT/TTS • Android Ready*

---

## Architecture Overview

```
[LIVE MIC / AUDIO FILE (WAV/PCM)]
              ↓
      [asrService] ──► [ASRModelAdapter (IndicConformer / Sherpa-ONNX)]
              ↓
      [SOURCE TRANSCRIPT (Hindi Text)]
              ↓
  [translationService] ──► [TranslationModelAdapter (IndicTrans2 INT8)]
              ↓
      [TARGET TRANSLATION (English Text)]
              ↓
      [ttsService] ──► [TTSModelAdapter (Piper/VITS Expressive-Ready)]
              ↓
       [SPEAKER AUDIO (WAV Out)]
```

Every service (`asrService`, `translationService`, `ttsService`, `modelManager`, `benchmarkService`, `audioService`, `languageService`) is completely modular and decoupled from the UI. Models are accessed through standard `BaseModelAdapter` interfaces (`load()`, `unload()`, `is_loaded()`, `get_metadata()`, `run_inference()`).

If any required model is not loaded, the system visibly displays **`"[CATEGORY] model not loaded"`** and halts execution cleanly. It will **never** silently fall back to a cloud service or browser speech recognition.

---

## Key Features

1. **Hardware Handheld Representation & Controls**:
   - Push-to-Talk (PTT) recording button
   - Audio wave visualizer canvas
   - Hardware LED status indicators (Offline Active, Processing, Standby)
   - Real-time 7-step pipeline visualizer:
     *Step 1: Audio captured → Step 2: ASR running → Step 3: Transcript produced → Step 4: Translation running → Step 5: Translation produced → Step 6: TTS running → Step 7: Audio ready*

2. **Model Manager**:
   - Manages local models for ASR, Translation, and TTS
   - Displays Name, Version, Size (MB), Format, Supported Languages, Status, RAM estimate, Load Time, and Inference Latency
   - Actions: **LOAD**, **UNLOAD**, **TEST**, **DELETE**, and **IMPORT LOCAL MODEL** (ONNX, ONNX-INT8, CTranslate2, TorchScript)

3. **Dedicated Benchmark Screen & Model Comparison**:
   - Exact Realtime Factor calculation:
     $$\text{Realtime Factor (RTF)} = \frac{\text{Processing Time (s)}}{\text{Audio Duration (s)}}$$
   - Color-coded performance badge:
     - 🟢 **Green (RTF < 0.70)**: Realtime execution
     - 🟡 **Yellow (0.70 ≤ RTF ≤ 1.20)**: Near-realtime execution
     - 🔴 **Red (RTF > 1.20)**: High latency
   - Head-to-head Model Comparison table running identical audio across multiple registered models
   - Precision latency breakdown ($T_{\text{ASR}}, T_{\text{Trans}}, T_{\text{TTS}}, T_{\text{Total}}$) and process RAM consumption via `psutil`

4. **10-Category Translation Quality Test (Prototype Evaluation)**:
   - Evaluates a 10-category benchmark dataset:
     1. Standard Hindi
     2. Conversational Hindi
     3. Hinglish / Code-switching (*"Bhai pharmacy kidhar hai? Mujhe medicine chahiye."*)
     4. Fast speech
     5. Accent variation
     6. Short phrases
     7. Longer sentence
     8. Numbers
     9. Names & Proper Nouns
     10. Everyday slang
   - Manual evaluation controls: **Correct**, **Mostly Correct**, **Needs Improvement**
   - Automatically computes aggregate **Prototype Evaluation Score** (clearly labeled as a prototype evaluation, not a formal academic benchmark)

5. **Expressive Prosody Architecture (Future Emotion Support)**:
   - The `ttsService.generate(text, language, style_params)` architecture natively accepts:
     - `speed`
     - `pitch`
     - `energy`
     - `style`
     - `emotion`
     - `expressiveness`
   - MVP defaults to `neutral`, allowing future prosody classifiers to feed directly into the TTS adapter without modifying the application layer.

6. **Airplane Mode Test**:
   - Disables and ignores all network connections
   - Enforces 100% on-device local execution
   - Network isolation audit verifying zero external requests

---

## Models Used in this POC

| Model Category | Model Name | Format | Primary Task | Model Storage Path |
| :--- | :--- | :--- | :--- | :--- |
| **ASR** | AI4Bharat IndicConformer (Hindi) | ONNX-INT8 | Hindi/Indic Speech Recognition | `models/asr/indic-conformer-hi-v1` |
| **ASR (Comparison)** | Whisper Tiny Indic (Mobile) | ONNX-INT8 | Compact ASR Comparison | `models/asr/whisper-tiny-indic` |
| **Translation** | AI4Bharat IndicTrans2 Distilled 200M | CTranslate2-INT8 | Indic $\leftrightarrow$ English Translation | `models/translation/indictrans2-hi-en-200m` |
| **Translation (Comp)** | AI4Bharat IndicTrans2 Indic-Indic 320M | CTranslate2-FP16 | Indic $\leftrightarrow$ Indic Translation | `models/translation/indictrans2-indic-indic-320m` |
| **TTS** | Piper / VITS Expressive-Ready | ONNX | Local Speech Synthesis | `models/tts/piper-vits-expressive` |
| **TTS (Target)** | AI4Bharat Indic Parler-TTS | ONNX-INT8 | Expressive Indic Synthesis | `models/tts/indic-parler-tts-exp` |

---

## Hardware Requirements

* **Development / Host PC**: Windows, Linux, or macOS. Python 3.12, 4 GB+ RAM.
* **Target Android Smartphone**:
  - RAM: 8 GB RAM recommended (6 GB minimum with INT8 quantization)
  - Processor: 64-bit ARM64-v8a (Snapdragon 7/8 Gen series, MediaTek Dimensity, Google Tensor)
  - Storage: ~800 MB free for local quantized weights

---

## Getting Started

### 1. Launch the Local Application
Run the one-click launcher:
```powershell
.\run_poc.bat
```
Or start via Python directly:
```powershell
& "C:\Users\HPS\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the Handheld UI
Open your browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Running the Automated Test Suite

To verify the offline pipeline, strict unloaded-model rejection, and RTF calculation:
```powershell
& "C:\Users\HPS\AppData\Local\Programs\Python\Python312\python.exe" -m unittest backend/tests/test_offline_pipeline.py
```

---

## How to Replace a Model Later

1. **Place Model Weights**:
   Export or download the new ONNX / CTranslate2 model files into the respective directory under `./models/{asr,translation,tts}/`.
2. **Register in Model Manager**:
   Go to the **Model Manager** tab in the UI and click **"+ IMPORT LOCAL MODEL"**, or update `backend/services/model_manager.py`.
3. **Run Benchmark**:
   Switch to the **Benchmark & Comparison** tab to verify that the new model loads, calculate its RAM footprint, and measure its Realtime Factor (RTF).

---

## Android Packaging Path

See [`ANDROID_PACKAGING_GUIDE.md`](file:///c:/Users/HPS/Desktop/ai%20speec%20reco/ANDROID_PACKAGING_GUIDE.md) for the complete production Android blueprint, including:
- `com.k2fsa.sherpa.onnx` Android AAR integration
- `com.microsoft.onnxruntime:onnxruntime-android` setup
- Omitting `android.permission.INTERNET` from `AndroidManifest.xml` to physically enforce offline operation.
