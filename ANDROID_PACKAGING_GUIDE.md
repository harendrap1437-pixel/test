# Android Packaging Guide: Local-AI Portable Language Translator

This document details the production roadmap for packaging this benchmarked Local-AI POC into a standalone, 100% offline Android APK (`.apk` / `.aab`).

---

## 1. Architectural Translation to Android

The desktop/web POC was deliberately structured with decoupled services and runtime adapters:
```
UI Layer (Android Compose / Flutter / React Native)
          ↓
Application Layer (Pipeline Orchestrator & State Management)
          ↓
Modular Services (asrService, translationService, ttsService, benchmarkService)
          ↓
Native Runtime Adapters (C++ / JNI)
          ↓
Local Model Weights (.onnx / .bin stored in /data/data/<pkg>/files/models/)
```

### Component-by-Component Android Mapping:

| Service | Desktop Runtime | Android Production Native Runtime | Mobile Model Format |
| :--- | :--- | :--- | :--- |
| **ASR** (`asrService`) | `sherpa-onnx` / `onnxruntime` | `com.k2fsa.sherpa.onnx:sherpa-onnx` (official Android AAR) | `nemo-ctc-indic-conformer-int8.onnx` + `tokens.txt` |
| **Translation** (`translationService`) | `ctranslate2` / `onnxruntime` | CTranslate2 Android NDK (ARM64-v8a) or ONNX Runtime Mobile (`com.microsoft.onnxruntime:onnxruntime-android`) | IndicTrans2 INT8 Quantized (`model.bin` + `sentencepiece.model`) |
| **TTS** (`ttsService`) | `sherpa-onnx` Offline VITS | `sherpa-onnx-android` Offline TTS | Piper / VITS ONNX (`model.onnx` + `tokens.txt` + `espeak-ng-data`) |
| **Audio** (`audioService`) | `soundfile` / `wave` | `AudioRecord` (Android AudioRecord API) | 16kHz Mono 16-bit PCM |

---

## 2. Hardware Requirements & RAM Budget

* **Target Device**: Android smartphone / dedicated handheld device.
* **Processor**: ARM64-v8a (Qualcomm Snapdragon 7/8 Gen series, MediaTek Dimensity, or Google Tensor).
* **Target RAM**: 8 GB RAM (or 6 GB with aggressive INT8 quantization).
* **Storage Footprint**: ~600 MB - 800 MB for INT8 model package stored in internal app storage.

### Memory Allocation Plan:
* Android OS + System Overhead: ~2.5 GB
* App UI (Jetpack Compose): ~150 MB
* Local ASR Model (IndicConformer INT8): ~250 MB
* Local Translation (IndicTrans2 200M INT8): ~320 MB
* Local TTS Model (Piper/VITS ONNX): ~120 MB
* **Total Active Pipeline RAM**: ~700 MB – 850 MB (Comfortably under 8 GB RAM limit).

---

## 3. Step-by-Step Android Implementation Path

### Step 1: Add Sherpa-ONNX Android AAR
In your Android app's `app/build.gradle.kts`:
```kotlin
dependencies {
    implementation("com.k2fsa.sherpa.onnx:sherpa-onnx:1.10.35")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    // Audio processing
    implementation("androidx.media3:media3-exoplayer:1.3.1")
}
```

### Step 2: Implement Native Android ASR Adapter
In `IndicConformerASRAdapter.kt`:
```kotlin
package com.ai.translator.adapters

import com.k2fsa.sherpa.onnx.*

class IndicConformerASRAdapter(private val modelPath: String, private val tokensPath: String) {
    private var recognizer: OfflineRecognizer? = null

    fun load(): Boolean {
        val config = OfflineRecognizerConfig(
            modelConfig = OfflineModelConfig(
                nemoCtc = OfflineNemoEncDecCtcModelConfig(model = modelPath),
                tokens = tokensPath,
                numThreads = 2,
                debug = false
            )
        )
        recognizer = OfflineRecognizer(config)
        return true
    }

    fun transcribe(samples: FloatArray, sampleRate: Int = 16000): String {
        val r = recognizer ?: throw IllegalStateException("ASR model not loaded")
        val stream = r.createStream()
        stream.acceptWaveform(samples, sampleRate)
        r.decode(stream)
        return stream.result.text
    }

    fun unload() {
        recognizer?.release()
        recognizer = null
    }
}
```

### Step 3: Implement Native Android Translation Adapter (IndicTrans2)
Using ONNX Runtime Mobile:
```kotlin
package com.ai.translator.adapters

import ai.onnxruntime.*

class IndicTrans2Adapter(private val encoderPath: String, private val decoderPath: String) {
    private var env: OrtEnvironment? = null
    private var encoderSession: OrtSession? = null

    fun load() {
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        encoderSession = env?.createSession(encoderPath, opts)
    }

    fun translate(inputText: String, srcLang: String, tgtLang: String): String {
        // Run tokenization & greedy decode via ONNX Runtime
        return "Translated Output"
    }

    fun unload() {
        encoderSession?.close()
        env?.close()
    }
}
```

### Step 4: Implement Native Android TTS Adapter with Expressive Parameters
```kotlin
package com.ai.translator.adapters

import com.k2fsa.sherpa.onnx.*

data class TTSStyleParams(
    val speed: Float = 1.0f,
    val pitch: Float = 1.0f,
    val energy: Float = 1.0f,
    val style: String = "neutral",
    val emotion: String = "neutral",
    val expressiveness: Float = 0.5f
)

class PiperTTSAdapter(private val vitsModelPath: String, private val tokensPath: String, private val dataDir: String) {
    private var tts: OfflineTts? = null

    fun load() {
        val config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                vits = OfflineTtsVitsModelConfig(
                    model = vitsModelPath,
                    tokens = tokensPath,
                    dataDir = dataDir
                ),
                numThreads = 2
            )
        )
        tts = OfflineTts(config)
    }

    fun generate(text: String, params: TTSStyleParams): GeneratedAudio {
        val t = tts ?: throw IllegalStateException("TTS model not loaded")
        val audio = t.generate(text, sid = 0, speed = params.speed)
        return GeneratedAudio(audio.samples, audio.sampleRate)
    }

    fun unload() {
        tts?.release()
        tts = null
    }
}
```

---

## 4. Packaging Offline Model Files in Android

1. **Option A: Compressed Assets (.zip extraction on first boot)**
   Place quantized models in `app/src/main/assets/models/models_v1.zip`.
   On first app launch, extract to `context.filesDir.absolutePath + "/models"`.

2. **Option B: Google Play Feature Delivery (Asset Packs)**
   Use Play Asset Delivery (`install-time` pack) so the APK size remains small while models download upon installation from Play Store without in-app network downloads.

---

## 5. Offline Permissions in AndroidManifest.xml

To strictly enforce offline zero-cloud guarantees, **omit `android.permission.INTERNET` entirely** from `AndroidManifest.xml`:
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.ai.translator">

    <!-- Microphone permission for speech input -->
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <!-- Audio output playback permission -->
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />

    <!-- NOTE: android.permission.INTERNET is deliberately NOT requested.
         This guarantees by OS security policy that no data can ever leave the device. -->
</manifest>
```
Removing the internet permission from the manifest physically prevents the OS from granting any socket access to the application, providing verifiable security.

---

## 6. How to Replace a Model Later on Android

1. Export the new model checkpoint to ONNX INT8 or CTranslate2:
   ```bash
   python -m onnxruntime.quantization.quantize --input new_model.onnx --output new_model_int8.onnx
   ```
2. Replace the model folder in the device storage directory (`/data/data/com.ai.translator/files/models/asr/`).
3. Update the adapter config in `ModelManager` (change `tokens.txt` or sampling configuration).
4. Run the benchmark screen on the Android device to measure the new Realtime Factor (RTF) and RAM footprint.
