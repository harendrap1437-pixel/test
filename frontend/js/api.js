/**
 * Local AI API Service Client.
 * Communicates ONLY with 127.0.0.1. Zero cloud or third-party connections.
 */
const API_BASE = window.location.origin;

export const api = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/api/status`);
    return await res.json();
  },

  async getLanguages() {
    const res = await fetch(`${API_BASE}/api/languages`);
    return await res.json();
  },

  async getSamples() {
    const res = await fetch(`${API_BASE}/api/samples`);
    return await res.json();
  },

  async translateAudioPipeline(formData) {
    const res = await fetch(`${API_BASE}/api/pipeline/translate-audio`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Pipeline translation failed");
    }
    return await res.json();
  },

  async translateText(text, srcLang, tgtLang) {
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
    const res = await fetch(`${API_BASE}/api/quality-test`);
    return await res.json();
  },

  async runQualityTest() {
    const res = await fetch(`${API_BASE}/api/quality-test/run`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Quality test run failed");
    }
    return await res.json();
  },

  async evaluateQualityItem(itemId, evaluation) {
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
