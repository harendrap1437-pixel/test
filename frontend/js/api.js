/**
 * Local AI API Service Client.
 * Communicates directly via in-app Native AndroidBridge or Local HTTP (127.0.0.1).
 * 100% Offline with zero cloud or third-party connections.
 */

const getApiBase = () => {
  if (typeof window !== "undefined" && window.location && window.location.origin && window.location.origin.startsWith("http")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8765";
};

const API_BASE = getApiBase();

const blobToBase64 = (blob) =>
  new Promise((resolve, reject) => {
    if (!blob) return resolve(null);
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

export const api = {
  async getStatus() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getStatus === "function") {
      try {
        const raw = window.AndroidBridge.getStatus();
        return JSON.parse(raw);
      } catch (e) {
        console.warn("AndroidBridge getStatus error, falling back to HTTP:", e);
      }
    }
    const res = await fetch(`${API_BASE}/api/status`);
    return await res.json();
  },

  async getLanguages() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getLanguages === "function") {
      try {
        const raw = window.AndroidBridge.getLanguages();
        return JSON.parse(raw);
      } catch (e) {
        console.warn("AndroidBridge getLanguages error, falling back to HTTP:", e);
      }
    }
    const res = await fetch(`${API_BASE}/api/languages`);
    return await res.json();
  },

  async getSamples() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getSamples === "function") {
      try {
        const raw = window.AndroidBridge.getSamples();
        return JSON.parse(raw);
      } catch (e) {
        console.warn("AndroidBridge getSamples error, falling back to HTTP:", e);
      }
    }
    const res = await fetch(`${API_BASE}/api/samples`);
    return await res.json();
  },

  async translateAudioPipeline(formData) {
    // Check if AndroidBridge direct native pipeline is available
    if (window.AndroidBridge && typeof window.AndroidBridge.runPipelineJson === "function") {
      try {
        const payload = {};
        for (const [key, value] of formData.entries()) {
          if (value instanceof Blob) {
            payload.audio_base64 = await blobToBase64(value);
          } else {
            payload[key] = value;
          }
        }
        const raw = window.AndroidBridge.runPipelineJson(JSON.stringify(payload));
        const res = JSON.parse(raw);
        if (res.error) throw new Error(res.error);
        return res;
      } catch (e) {
        console.warn("AndroidBridge runPipelineJson error, attempting HTTP fallback:", e);
      }
    }

    try {
      const res = await fetch(`${API_BASE}/api/pipeline/translate-audio`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Pipeline translation failed" }));
        throw new Error(err.detail || `Server returned ${res.status}`);
      }
      return await res.json();
    } catch (fetchErr) {
      // If HTTP fails on Android, fallback to Bridge
      if (window.AndroidBridge && typeof window.AndroidBridge.runPipelineJson === "function") {
        const payload = {};
        for (const [key, value] of formData.entries()) {
          if (value instanceof Blob) {
            payload.audio_base64 = await blobToBase64(value);
          } else {
            payload[key] = value;
          }
        }
        const raw = window.AndroidBridge.runPipelineJson(JSON.stringify(payload));
        return JSON.parse(raw);
      }
      throw fetchErr;
    }
  },

  async translateText(text, srcLang, tgtLang) {
    if (window.AndroidBridge && typeof window.AndroidBridge.translateText === "function") {
      try {
        const raw = window.AndroidBridge.translateText(text, srcLang, tgtLang);
        return JSON.parse(raw);
      } catch (e) {
        console.warn("AndroidBridge translateText fallback:", e);
      }
    }
    const res = await fetch(`${API_BASE}/api/pipeline/translate-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, src_lang: srcLang, tgt_lang: tgtLang })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Translation failed");
    }
    return await res.json();
  },

  async generateTTS(text, language, styleParams = {}) {
    const res = await fetch(`${API_BASE}/api/pipeline/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language,
        speed: styleParams.speed ?? 1.0,
        pitch: styleParams.pitch ?? 1.0,
        energy: styleParams.energy ?? 1.0,
        style: styleParams.style ?? "neutral",
        emotion: styleParams.emotion ?? "neutral",
        expressiveness: styleParams.expressiveness ?? 0.5
      })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "TTS generation failed");
    }
    return await res.json();
  },

  async getModels() {
    const res = await fetch(`${API_BASE}/api/models`);
    return await res.json();
  },

  async activateModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/${category}/${modelId}/activate`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Model activation failed");
    }
    return await res.json();
  },

  async loadModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/${category}/${modelId}/load`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Model load failed");
    }
    return await res.json();
  },

  async unloadModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/${category}/${modelId}/unload`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Model unload failed");
    }
    return await res.json();
  },

  async testModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/${category}/${modelId}/test`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Model test failed");
    }
    return await res.json();
  },

  async deleteModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/${category}/${modelId}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Model delete failed");
    }
    return await res.json();
  },

  async importModel(payload) {
    const res = await fetch(`${API_BASE}/api/models/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Import model failed");
    }
    return await res.json();
  },

  async getBenchmarkHistory() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getBenchmarkHistory === "function") {
      try {
        const raw = window.AndroidBridge.getBenchmarkHistory();
        return JSON.parse(raw);
      } catch (e) {}
    }
    const res = await fetch(`${API_BASE}/api/benchmark/history`);
    return await res.json();
  },

  async compareModels(sampleFilename = null) {
    const url = sampleFilename
      ? `${API_BASE}/api/benchmark/compare?sample_filename=${encodeURIComponent(sampleFilename)}`
      : `${API_BASE}/api/benchmark/compare`;
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Comparison failed");
    }
    return await res.json();
  },

  async getQualityTestDataset() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getQualityDataset === "function") {
      try {
        const raw = window.AndroidBridge.getQualityDataset();
        return JSON.parse(raw);
      } catch (e) {}
    }
    const res = await fetch(`${API_BASE}/api/quality-test`);
    return await res.json();
  },

  async runQualityTest() {
    if (window.AndroidBridge && typeof window.AndroidBridge.runQualityTest === "function") {
      try {
        const raw = window.AndroidBridge.runQualityTest();
        return JSON.parse(raw);
      } catch (e) {}
    }
    const res = await fetch(`${API_BASE}/api/quality-test/run`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Quality test run failed");
    }
    return await res.json();
  },

  async evaluateQualityItem(itemId, evaluation) {
    if (window.AndroidBridge && typeof window.AndroidBridge.evaluateQualityItem === "function") {
      try {
        const raw = window.AndroidBridge.evaluateQualityItem(itemId, evaluation);
        return JSON.parse(raw);
      } catch (e) {}
    }
    const res = await fetch(`${API_BASE}/api/quality-test/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: itemId, evaluation })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Evaluation failed");
    }
    return await res.json();
  },

  async analyzeEmotion(formData) {
    if (window.AndroidBridge && typeof window.AndroidBridge.analyzeEmotion === "function") {
      try {
        const payload = {};
        for (const [key, value] of formData.entries()) {
          if (value instanceof Blob) {
            payload.audio_base64 = await blobToBase64(value);
          } else {
            payload[key] = value;
          }
        }
        const raw = window.AndroidBridge.analyzeEmotion(JSON.stringify(payload));
        return JSON.parse(raw);
      } catch (e) {
        console.warn("AndroidBridge analyzeEmotion fallback:", e);
      }
    }
    const res = await fetch(`${API_BASE}/api/emotion/analyze-audio`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Emotion analysis failed");
    }
    return await res.json();
  },

  async getEmotionSamples() {
    if (window.AndroidBridge && typeof window.AndroidBridge.getEmotionSamples === "function") {
      try {
        const raw = window.AndroidBridge.getEmotionSamples();
        return JSON.parse(raw);
      } catch (e) {}
    }
    const res = await fetch(`${API_BASE}/api/emotion/samples`);
    return await res.json();
  },

  async getOfflineStatus() {
    const res = await fetch(`${API_BASE}/api/offline/status`);
    return await res.json();
  },

  async toggleOffline(enabled) {
    const res = await fetch(`${API_BASE}/api/offline/toggle?enabled=${enabled}`, { method: "POST" });
    return await res.json();
  }
};

