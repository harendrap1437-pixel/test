import { api } from "./api.js";

export class TranslatorController {
  constructor(handheld) {
    this.handheld = handheld;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.recordedAudioBlob = null;
    this.selectedSample = null;
    this.currentAudioOutputUrl = null;
    this.audioElement = new Audio();

    this.inputMode = "mic"; // "mic" or "file"

    this.initElements();
    this.bindEvents();
    this.loadLanguages();
    this.loadSamples();
    this.updateActiveModelLabels();
  }

  initElements() {
    this.srcLangSelect = document.getElementById("srcLangSelect");
    this.tgtLangSelect = document.getElementById("tgtLangSelect");
    this.swapLangBtn = document.getElementById("swapLangBtn");
    this.pttBtn = document.getElementById("pttMainBtn");
    this.translateBtn = document.getElementById("translateActionBtn");
    this.replayBtn = document.getElementById("replayAudioBtn");
    this.stopAudioBtn = document.getElementById("stopAudioBtn");

    this.lcdSourceText = document.getElementById("lcdSourceText");
    this.lcdTargetText = document.getElementById("lcdTargetText");
    this.lcdEmotionBadge = document.getElementById("lcdEmotionBadge");
    this.audioDurationTag = document.getElementById("audioDurationTag");
    this.pipelineErrorMessage = document.getElementById("pipelineErrorMessage");

    // Stepper nodes (8 modular stages)
    this.steps = {
      1: document.getElementById("stepNode1"),
      2: document.getElementById("stepNode2"),
      3: document.getElementById("stepNode3"),
      4: document.getElementById("stepNode4"),
      5: document.getElementById("stepNode5"),
      6: document.getElementById("stepNode6"),
      7: document.getElementById("stepNode7"),
      8: document.getElementById("stepNode8")
    };
    this.stepEmotionResult = document.getElementById("stepEmotionResult");

    // Expressive parameters
    this.prosodySpeed = document.getElementById("prosodySpeed");
    this.prosodyPitch = document.getElementById("prosodyPitch");
    this.prosodyEnergy = document.getElementById("prosodyEnergy");
    this.prosodyEmotion = document.getElementById("prosodyEmotion");
    this.prosodyExpressiveness = document.getElementById("prosodyExpressiveness");

    // Mode toggles
    this.modeMicBtn = document.getElementById("modeMicBtn");
    this.modeFileBtn = document.getElementById("modeFileBtn");
    this.micInputSection = document.getElementById("micInputSection");
    this.fileInputSection = document.getElementById("fileInputSection");
    this.audioFileInput = document.getElementById("audioFileInput");
    this.samplePillsContainer = document.getElementById("samplePillsContainer");

    // Standalone Emotion SER elements
    this.btnAnalyzeEmotion = document.getElementById("btnAnalyzeEmotion");
    this.emotionStatusTag = document.getElementById("emotionStatusTag");
    this.emotionBarsGrid = document.getElementById("emotionBarsGrid");
    this.selectedEmotionSample = "sample_hi_neutral.wav";
  }

  bindEvents() {
    this.pttBtn.addEventListener("click", () => this.toggleLiveRecording());
    this.translateBtn.addEventListener("click", () => this.executeTranslationPipeline());
    this.replayBtn.addEventListener("click", () => this.playAudio());
    this.stopAudioBtn.addEventListener("click", () => this.stopAudio());

    // Standalone Emotion SER analyzer events
    if (this.btnAnalyzeEmotion) {
      this.btnAnalyzeEmotion.addEventListener("click", () => this.runStandaloneEmotionAnalysis());
    }

    document.querySelectorAll(".emotion-sample-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".emotion-sample-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.selectedEmotionSample = btn.dataset.sample;
        if (this.emotionStatusTag) {
          this.emotionStatusTag.textContent = `Selected: ${btn.dataset.sample}. Click Run Inference.`;
        }
      });
    });

    this.swapLangBtn.addEventListener("click", () => {
      const temp = this.srcLangSelect.value;
      this.srcLangSelect.value = this.tgtLangSelect.value;
      this.tgtLangSelect.value = temp;
    });

    this.modeMicBtn.addEventListener("click", () => this.setInputMode("mic"));
    this.modeFileBtn.addEventListener("click", () => this.setInputMode("file"));

    this.audioFileInput.addEventListener("change", async (e) => {
      if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        this.selectedSample = null;
        this.audioDurationTag.textContent = `Processing ${file.name}...`;
        this.translateBtn.disabled = true;
        this.resetPipelineSteps();
        this.markStep(1, "active");

        try {
          this.recordedAudioBlob = await this.convertTo16kHzMonoWav(file);
          this.audioDurationTag.textContent = `${file.name} (16kHz WAV Ready)`;
        } catch (err) {
          console.warn("Client audio conversion fallback to raw file:", err);
          this.recordedAudioBlob = file;
          this.audioDurationTag.textContent = `${file.name}`;
        }
        this.translateBtn.disabled = false;
        this.markStep(1, "done");
      }
    });

    this.audioElement.addEventListener("ended", () => {
      this.handheld.stopWaveform();
      this.replayBtn.disabled = false;
    });

    window.addEventListener("models-updated", () => this.updateActiveModelLabels());
  }

  async updateActiveModelLabels() {
    try {
      const modelsData = await api.getModels();
      const asr = (modelsData.asr || []).find((m) => m.is_active && m.is_loaded) || (modelsData.asr || []).find((m) => m.is_loaded) || (modelsData.asr || [])[0];
      const trans = (modelsData.translation || []).find((m) => m.is_active && m.is_loaded) || (modelsData.translation || []).find((m) => m.is_loaded) || (modelsData.translation || [])[0];
      const tts = (modelsData.tts || []).find((m) => m.is_active && m.is_loaded) || (modelsData.tts || []).find((m) => m.is_loaded) || (modelsData.tts || [])[0];

      if (asr) {
        const asrName = `${asr.name}`;
        const mAsr = document.getElementById("matrixAsrName");
        const sAsr = document.getElementById("stepAsrName");
        const lAsr = document.getElementById("lcdAsrBadge");
        if (mAsr) mAsr.textContent = asrName;
        if (sAsr) sAsr.textContent = asrName;
        if (lAsr) lAsr.textContent = `ASR: ${asr.id}`;
      }

      if (trans) {
        const transName = `${trans.name}`;
        const mTrans = document.getElementById("matrixTransName");
        const sTrans = document.getElementById("stepTransName");
        const lTrans = document.getElementById("lcdTransBadge");
        if (mTrans) mTrans.textContent = transName;
        if (sTrans) sTrans.textContent = transName;
        if (lTrans) lTrans.textContent = `MT: ${trans.id}`;
      }

      if (tts) {
        const ttsName = `${tts.name}`;
        const mTts = document.getElementById("matrixTtsName");
        const sTts = document.getElementById("stepTtsName");
        if (mTts) mTts.textContent = ttsName;
        if (sTts) sTts.textContent = ttsName;
      }
    } catch (e) {
      console.warn("Could not fetch active models for translator:", e);
    }
  }

  setInputMode(mode) {
    this.inputMode = mode;
    if (mode === "mic") {
      this.modeMicBtn.classList.add("active");
      this.modeFileBtn.classList.remove("active");
      this.micInputSection.style.display = "block";
      this.fileInputSection.style.display = "none";
    } else {
      this.modeMicBtn.classList.remove("active");
      this.modeFileBtn.classList.add("active");
      this.micInputSection.style.display = "none";
      this.fileInputSection.style.display = "block";
    }
  }

  async loadLanguages() {
    try {
      const langs = await api.getLanguages();
      this.srcLangSelect.innerHTML = "";
      this.tgtLangSelect.innerHTML = "";
      langs.forEach((l) => {
        const opt1 = new Option(`${l.name} (${l.native})`, l.code);
        const opt2 = new Option(`${l.name} (${l.native})`, l.code);
        this.srcLangSelect.add(opt1);
        this.tgtLangSelect.add(opt2);
      });
      this.srcLangSelect.value = "hi";
      this.tgtLangSelect.value = "en";
    } catch (e) {
      console.error("Failed to load languages:", e);
    }
  }

  async loadSamples() {
    try {
      const samples = await api.getSamples();
      this.samplePillsContainer.innerHTML = "";
      samples.forEach((s) => {
        const btn = document.createElement("button");
        btn.className = "sample-audio-btn";
        btn.textContent = `▶ ${s.filename} (${s.duration_sec}s)`;
        btn.onclick = () => {
          this.selectedSample = s.filename;
          this.recordedAudioBlob = null;
          this.audioDurationTag.textContent = `${s.filename} [${s.duration_sec}s]`;
          this.translateBtn.disabled = false;
          this.resetPipelineSteps();
          this.markStep(1, "done");
          this.lcdSourceText.textContent = `Selected: ${s.filename}`;
        };
        this.samplePillsContainer.appendChild(btn);
      });
    } catch (e) {
      console.error("Failed to load audio samples:", e);
    }
  }

  async convertTo16kHzMonoWav(rawBlob) {
    try {
      const arrayBuffer = await rawBlob.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      // Resample to 16,000 Hz mono (Whisper & Conformer standard)
      const targetSampleRate = 16000;
      const offlineCtx = new OfflineAudioContext(1, Math.max(1, Math.ceil(decodedBuffer.duration * targetSampleRate)), targetSampleRate);
      const source = offlineCtx.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(offlineCtx.destination);
      source.start(0);

      const renderedBuffer = await offlineCtx.startRendering();
      const channelData = renderedBuffer.getChannelData(0);

      // Encode into standard 16-bit mono RIFF WAV
      const wavBuffer = new ArrayBuffer(44 + channelData.length * 2);
      const view = new DataView(wavBuffer);

      const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };

      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + channelData.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true); // Subchunk1Size
      view.setUint16(20, 1, true);  // AudioFormat: PCM
      view.setUint16(22, 1, true);  // NumChannels: 1
      view.setUint32(24, targetSampleRate, true);
      view.setUint32(28, targetSampleRate * 2, true);
      view.setUint16(32, 2, true);  // BlockAlign
      view.setUint16(34, 16, true); // BitsPerSample
      writeString(view, 36, 'data');
      view.setUint32(40, channelData.length * 2, true);

      let offset = 44;
      for (let i = 0; i < channelData.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      return new Blob([wavBuffer], { type: 'audio/wav' });
    } catch (e) {
      console.warn("Local WAV conversion fallback to raw blob:", e);
      return rawBlob;
    }
  }

  async toggleLiveRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };

      this.mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType || "audio/webm" });
        this.audioDurationTag.textContent = "Processing audio...";
        this.recordedAudioBlob = await this.convertTo16kHzMonoWav(rawBlob);
        this.selectedSample = null;
        this.audioDurationTag.textContent = "Mic Audio Ready (16kHz WAV)";
        this.translateBtn.disabled = false;
        this.markStep(1, "done");
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.pttBtn.classList.add("recording");
      this.pttBtn.innerHTML = `<span>⏹ STOP RECORDING</span>`;
      this.handheld.startWaveform(true);
      this.resetPipelineSteps();
      this.markStep(1, "active");
      this.hideError();
    } catch (err) {
      this.showError(`Microphone access error: ${err.message}. You can also use Local Audio File / Presets.`);
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach((t) => t.stop());
      this.isRecording = false;
      this.pttBtn.classList.remove("recording");
      this.pttBtn.innerHTML = `<span>🎙 PUSH TO TALK (MIC)</span>`;
      this.handheld.stopWaveform();
    }
  }

  async executeTranslationPipeline() {
    this.hideError();
    this.resetPipelineSteps();
    this.markStep(1, "done");
    this.markStep(2, "active");
    this.handheld.startWaveform(false);

    const formData = new FormData();
    formData.append("src_lang", this.srcLangSelect.value);
    formData.append("tgt_lang", this.tgtLangSelect.value);

    // Style/Prosody parameters for Expressive TTS
    formData.append("speed", this.prosodySpeed ? this.prosodySpeed.value : 1.0);
    formData.append("pitch", this.prosodyPitch ? this.prosodyPitch.value : 1.0);
    formData.append("energy", this.prosodyEnergy ? this.prosodyEnergy.value : 1.0);
    formData.append("emotion", this.prosodyEmotion ? this.prosodyEmotion.value : "neutral");
    formData.append("expressiveness", this.prosodyExpressiveness ? this.prosodyExpressiveness.value : 0.5);

    if (this.selectedSample) {
      formData.append("sample_filename", this.selectedSample);
    } else if (this.recordedAudioBlob) {
      formData.append("audio_file", this.recordedAudioBlob, "audio_input.wav");
    } else {
      // Default to sample
      formData.append("sample_filename", "sample_hi_phr1_pharmacy.wav");
    }

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    try {
      // Hardware LEDs: Blue ON (Processing Active)
      this.handheld.setLed("blue", true);
      this.handheld.setLed("green", true);
      this.handheld.setLed("amber", false);

      // Visual Stage 1: Audio Captured Done
      this.markStep(1, "done");

      // Visual Stage 2: ASR Running
      this.markStep(2, "active");
      this.lcdSourceText.innerHTML = `<span class="pipeline-pulse-text">⚡ Step 2: Running local ASR inference...</span>`;
      this.lcdTargetText.innerHTML = `<span style="color:var(--text-muted); font-style:italic;">Waiting for transcript...</span>`;

      // Call local modular pipeline API
      const res = await api.translateAudioPipeline(formData);

      // Visual Stage 2 Complete
      this.markStep(2, "done");
      await sleep(150);

      // Visual Stage 3: Transcript Produced
      this.markStep(3, "done");
      this.lcdSourceText.textContent = res.source_transcript;
      await sleep(150);

      // Visual Stage 4: Emotion SER Analysis (Real Acoustic Waveform Evaluation)
      this.markStep(4, "active");
      const emoStr = (res.detected_emotion || "neutral").toUpperCase();
      const emoConfPct = (res.emotion_confidence ? res.emotion_confidence * 100 : 85).toFixed(1);
      if (this.stepEmotionResult) {
        this.stepEmotionResult.textContent = `${emoStr} (${emoConfPct}%)`;
      }
      if (this.lcdEmotionBadge) {
        this.lcdEmotionBadge.textContent = `EMOTION: ${emoStr} (${emoConfPct}%)`;
        this.lcdEmotionBadge.style.color = emoStr === "ANGRY" ? "#ff5252" : (emoStr === "HAPPY" ? "#69f0ae" : (emoStr === "SAD" ? "#40c4ff" : "#ffb74d"));
      }
      await sleep(200);
      this.markStep(4, "done");

      // Visual Stage 5: Translation Running
      this.markStep(5, "active");
      this.lcdTargetText.innerHTML = `<span class="pipeline-pulse-text">⚡ Step 5: Translating with IndicTrans2...</span>`;
      await sleep(200);

      // Visual Stage 6: Translation Produced
      this.markStep(5, "done");
      this.markStep(6, "done");
      this.lcdTargetText.textContent = res.translated_text;
      await sleep(180);

      // Visual Stage 7: Emotion-Aware TTS Running
      this.markStep(7, "active");
      await sleep(200);

      // Visual Stage 8: Audio Output Ready
      this.markStep(7, "done");
      this.markStep(8, "done");

      this.currentAudioOutputUrl = res.audio_base64;
      this.audioElement.src = this.currentAudioOutputUrl;
      this.replayBtn.disabled = false;
      this.stopAudioBtn.disabled = false;

      // Hardware LEDs: Processing complete
      this.handheld.setLed("blue", false);

      if (res.source_transcript === "(No speech detected)") {
        this.showNotice("ℹ️ No clear speech detected. Speak clearly into the microphone or test with one of the sample buttons below.");
      } else {
        // Automatically play output audio
        this.playAudio();
      }

      // Trigger benchmark history refresh
      window.dispatchEvent(new CustomEvent("benchmark-updated", { detail: res.benchmark }));
    } catch (err) {
      this.showError(err.message);
      this.handheld.setLed("amber", true);
      this.handheld.setLed("blue", false);
    } finally {
      this.handheld.stopWaveform();
    }
  }

  async runStandaloneEmotionAnalysis() {
    if (!this.emotionBarsGrid || !this.btnAnalyzeEmotion) return;
    this.btnAnalyzeEmotion.disabled = true;
    this.btnAnalyzeEmotion.textContent = "⏳ Running emotion2vec+...";
    if (this.emotionStatusTag) {
      this.emotionStatusTag.textContent = `Analyzing ${this.selectedEmotionSample || "recorded audio"}...`;
    }

    try {
      const formData = new FormData();
      if (this.recordedAudioBlob) {
        formData.append("audio_file", this.recordedAudioBlob, "audio.wav");
      } else {
        formData.append("sample_filename", this.selectedEmotionSample || "sample_hi_neutral.wav");
      }

      const res = await api.analyzeEmotion(formData);
      if (this.emotionStatusTag) {
        this.emotionStatusTag.innerHTML = `Predicted Emotion: <strong style="color:var(--color-amber);">${res.emotion.toUpperCase()}</strong> (${(res.confidence * 100).toFixed(1)}%) in <strong>${res.latency_ms} ms</strong>`;
      }

      // Render score meters for all 9 emotion classes
      this.emotionBarsGrid.innerHTML = "";
      const sortedScores = Object.entries(res.scores || {}).sort((a, b) => b[1] - a[1]);
      
      sortedScores.forEach(([emo, score]) => {
        const pct = (score * 100).toFixed(1);
        const barItem = document.createElement("div");
        barItem.style.cssText = "background:rgba(0,0,0,0.4); padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06);";
        const isTop = emo === res.emotion;
        const color = isTop ? "var(--color-amber)" : (score > 0.1 ? "var(--accent-cyan)" : "var(--text-muted)");
        
        barItem.innerHTML = `
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px;">
            <span style="font-weight:${isTop ? '700' : 'normal'}; color:${color};">${emo.toUpperCase()}</span>
            <span style="font-weight:700; color:${color};">${pct}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden;">
            <div style="background:${color}; width:${pct}%; height:100%; border-radius:3px; transition:width 0.4s ease;"></div>
          </div>
        `;
        this.emotionBarsGrid.appendChild(barItem);
      });

    } catch (e) {
      if (this.emotionStatusTag) {
        this.emotionStatusTag.textContent = `❌ Error: ${e.message}`;
      }
    } finally {
      this.btnAnalyzeEmotion.disabled = false;
      this.btnAnalyzeEmotion.textContent = "⚡ Run emotion2vec+ Inference";
    }
  }

  playAudio() {
    if (this.currentAudioOutputUrl) {
      this.audioElement.play();
      this.handheld.startWaveform(false);
      this.replayBtn.disabled = true;
    }
  }

  stopAudio() {
    this.audioElement.pause();
    this.audioElement.currentTime = 0;
    this.handheld.stopWaveform();
    this.replayBtn.disabled = false;
  }

  resetPipelineSteps() {
    for (let i = 1; i <= 8; i++) {
      if (this.steps[i]) {
        this.steps[i].classList.remove("active", "done");
      }
    }
  }

  markStep(stepNum, status) {
    if (this.steps[stepNum]) {
      this.steps[stepNum].classList.remove("active", "done");
      this.steps[stepNum].classList.add(status);
    }
  }

  showNotice(msg) {
    if (this.pipelineErrorMessage) {
      this.pipelineErrorMessage.textContent = msg;
      this.pipelineErrorMessage.style.display = "block";
      this.pipelineErrorMessage.style.borderColor = "rgba(0, 229, 255, 0.4)";
      this.pipelineErrorMessage.style.color = "#80d8ff";
      this.pipelineErrorMessage.style.background = "rgba(0, 229, 255, 0.08)";
    }
  }

  showError(msg) {
    if (this.pipelineErrorMessage) {
      this.pipelineErrorMessage.textContent = `❌ ${msg}`;
      this.pipelineErrorMessage.style.display = "block";
      this.pipelineErrorMessage.style.borderColor = "#f43f5e";
      this.pipelineErrorMessage.style.color = "#ff80ab";
      this.pipelineErrorMessage.style.background = "rgba(244, 63, 94, 0.12)";
    }
  }

  hideError() {
    if (this.pipelineErrorMessage) {
      this.pipelineErrorMessage.style.display = "none";
      this.pipelineErrorMessage.textContent = "";
    }
  }
}
