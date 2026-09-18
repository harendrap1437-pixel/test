import { api } from "./api.js";

export class ModelManagerController {
  constructor() {
    this.container = document.getElementById("modelsContainer");
    this.importModal = document.getElementById("importModelModal");
    this.btnOpenImport = document.getElementById("btnOpenImportModel");
    this.btnCloseImport = document.getElementById("btnCloseImportModal");
    this.btnSubmitImport = document.getElementById("btnSubmitImportModel");

    this.bindEvents();
    this.loadModels();
  }

  bindEvents() {
    if (this.btnOpenImport) {
      this.btnOpenImport.addEventListener("click", () => {
        this.importModal.classList.add("open");
      });
    }
    if (this.btnCloseImport) {
      this.btnCloseImport.addEventListener("click", () => {
        this.importModal.classList.remove("open");
      });
    }
    if (this.btnSubmitImport) {
      this.btnSubmitImport.addEventListener("click", () => this.handleImport());
    }

    const importFileInput = document.getElementById("importModelFile");
    if (importFileInput) {
      importFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          const nameInput = document.getElementById("importName");
          const sizeInput = document.getElementById("importSize");
          if (nameInput && (!nameInput.value || nameInput.value === "")) {
            nameInput.value = file.name.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
          }
          if (sizeInput) {
            sizeInput.value = (file.size / (1024 * 1024)).toFixed(1);
          }
        }
      });
    }
  }

  async handleImport() {
    const category = document.getElementById("importCategory").value;
    const nameInput = document.getElementById("importName");
    const name = nameInput ? nameInput.value : "";
    const version = document.getElementById("importVersion").value || "1.0";
    const format = document.getElementById("importFormat").value || "ONNX-INT8";
    const size_mb = parseFloat(document.getElementById("importSize").value) || 150.0;
    const langs = document.getElementById("importLangs").value.split(",").map((s) => s.trim());

    if (!name) {
      alert("Please provide a model name or select a model file.");
      return;
    }

    try {
      await api.importModel({
        category,
        name,
        version,
        format,
        size_mb,
        supported_languages: langs
      });
      this.importModal.classList.remove("open");
      this.loadModels();
    } catch (e) {
      alert(e.message);
    }
  }

  async loadModels() {
    try {
      const data = await api.getModels();
      this.renderCategories(data);
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }

  renderCategories(data) {
    this.container.innerHTML = "";

    const categories = [
      { key: "asr", label: "🎙 Automatic Speech Recognition (ASR) Models", desc: "IndicConformer / Sherpa-ONNX / Whisper INT8" },
      { key: "emotion", label: "🧠 Speech Emotion Recognition (SER) Models", desc: "emotion2vec+ Base Distilled ONNX (9.25 MB) - 16kHz Raw Audio" },
      { key: "translation", label: "🌐 Local Translation Models", desc: "AI4Bharat IndicTrans2 Distilled 200M/320M CTranslate2/ONNX" },
      { key: "tts", label: "🔊 Local Text-To-Speech (TTS) Models", desc: "Piper / VITS / Indic Parler with Expressive Prosody Modulation" }
    ];

    categories.forEach((cat) => {
      const catSection = document.createElement("div");
      catSection.className = "model-category-section";

      const header = document.createElement("div");
      header.className = "model-category-header";
      header.innerHTML = `<span>${cat.label}</span><span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">${cat.desc}</span>`;
      catSection.appendChild(header);

      const grid = document.createElement("div");
      grid.className = "models-grid";

      const models = data[cat.key] || [];
      if (models.length === 0) {
        grid.innerHTML = `<p style="color:var(--text-muted); font-size:0.85rem;">No local models found in this category.</p>`;
      } else {
        models.forEach((m) => {
          grid.appendChild(this.createModelCard(cat.key, m));
        });
      }

      catSection.appendChild(grid);
      this.container.appendChild(catSection);
    });
  }

  createModelCard(category, m) {
    const isLoaded = m.is_loaded ?? m.loaded ?? false;
    const isActive = m.is_active ?? false;
    const supportedLangs = m.supported_languages || m.languages || ["hi", "en"];
    const ramEst = m.ram_estimate_mb || (m.size_mb ? Math.round(m.size_mb * 1.4) : 100);
    const statusText = m.status || (isLoaded ? "Loaded & Ready" : "Not Loaded");
    const formatText = m.format || "ONNX-INT8";
    const versionText = m.version || "1.0";

    const card = document.createElement("div");
    card.className = `model-card ${isLoaded ? "loaded" : ""}`;

    const statusBadge = isLoaded
      ? `<span class="model-badge badge-loaded">● LOADED</span>`
      : `<span class="model-badge badge-unloaded">○ NOT LOADED</span>`;

    const activeIndicator = isActive
      ? `<span style="background:rgba(0,230,118,0.2); color:#00e676; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:800; border:1px solid rgba(0,230,118,0.4); margin-left:8px;">★ ACTIVE DEFAULT</span>`
      : "";

    card.innerHTML = `
      <div class="model-card-header">
        <div>
          <div class="model-title">${m.name || m.id} ${activeIndicator}</div>
          <span style="font-size:0.75rem; color:var(--text-muted);">${m.id} (v${versionText})</span>
        </div>
        ${statusBadge}
      </div>

      <div class="model-meta-grid">
        <div class="meta-field">
          <span class="label">Format</span>
          <span class="val">${formatText}</span>
        </div>
        <div class="meta-field">
          <span class="label">Disk Size</span>
          <span class="val">${m.size_mb || 100} MB</span>
        </div>
        <div class="meta-field">
          <span class="label">RAM Est.</span>
          <span class="val">~${ramEst} MB</span>
        </div>
        <div class="meta-field">
          <span class="label">Languages</span>
          <span class="val">${Array.isArray(supportedLangs) ? supportedLangs.join(", ") : supportedLangs}</span>
        </div>
        <div class="meta-field">
          <span class="label">Load Time</span>
          <span class="val">${m.load_time_ms ? m.load_time_ms + " ms" : "--"}</span>
        </div>
        <div class="meta-field">
          <span class="label">Last Latency</span>
          <span class="val">${m.inference_time_ms ? m.inference_time_ms + " ms" : "--"}</span>
        </div>
      </div>

      <div style="font-size:0.75rem; color:var(--text-secondary); background:rgba(0,0,0,0.3); padding:6px 10px; border-radius:4px;">
        Status: <strong style="color:#fff;">${statusText}</strong>
      </div>

      <div class="model-card-actions" style="display:flex; flex-wrap:wrap; gap:6px; margin-top:10px;">
        ${
          isActive
            ? `<button class="btn-model-action" disabled style="background:#00e676; color:#050b14; font-weight:800; border:none; cursor:default; padding:6px 12px; border-radius:4px;">✓ ACTIVE DEFAULT</button>`
            : `<button class="btn-model-action" data-action="activate" style="border-color:var(--accent-cyan); color:var(--accent-cyan); font-weight:700; padding:6px 12px; border-radius:4px; cursor:pointer;">★ SET AS DEFAULT</button>`
        }
        ${
          isLoaded
            ? `<button class="btn-model-action unload" data-action="unload" style="cursor:pointer;">UNLOAD</button>`
            : `<button class="btn-model-action load" data-action="load" style="cursor:pointer;">LOAD</button>`
        }
        <button class="btn-model-action" data-action="test" style="cursor:pointer;">TEST</button>
        <button class="btn-model-action" data-action="delete" style="color:var(--color-red); border-color:#3b1e25; cursor:pointer;">DELETE</button>
      </div>
    `;

    // Action button events
    card.querySelector('[data-action="activate"]')?.addEventListener("click", async () => {
      try {
        await api.activateModel(category, m.id);
        await this.loadModels();
        window.dispatchEvent(new CustomEvent("models-updated"));
      } catch (e) {
        alert(e.message);
      }
    });

    card.querySelector('[data-action="load"]')?.addEventListener("click", async () => {
      try {
        await api.activateModel(category, m.id);
        await this.loadModels();
        window.dispatchEvent(new CustomEvent("models-updated"));
      } catch (e) {
        alert(e.message);
      }
    });

    card.querySelector('[data-action="unload"]')?.addEventListener("click", async () => {
      try {
        await api.unloadModel(category, m.id);
        await this.loadModels();
        window.dispatchEvent(new CustomEvent("models-updated"));
      } catch (e) {
        alert(e.message);
      }
    });

    card.querySelector('[data-action="test"]')?.addEventListener("click", async () => {
      try {
        const res = await api.testModel(category, m.id);
        alert(`Model Test Successful!\nLatency: ${res.test_latency_ms} ms\nStatus: ${res.status}`);
        this.loadModels();
      } catch (e) {
        alert(`Test Failed: ${e.message}`);
      }
    });

    card.querySelector('[data-action="delete"]')?.addEventListener("click", async () => {
      if (confirm(`Are you sure you want to delete ${m.name}?`)) {
        try {
          await api.deleteModel(category, m.id);
          this.loadModels();
        } catch (e) {
          alert(e.message);
        }
      }
    });

    return card;
  }

  async handleImport() {
    const category = document.getElementById("importCategory").value;
    const name = document.getElementById("importName").value;
    const version = document.getElementById("importVersion").value;
    const format = document.getElementById("importFormat").value;
    const size_mb = parseFloat(document.getElementById("importSize").value) || 150.0;
    const langs = document.getElementById("importLangs").value.split(",").map((s) => s.trim());

    if (!name) {
      alert("Please provide a model name.");
      return;
    }

    try {
      await api.importModel({
        category,
        name,
        version,
        format,
        size_mb,
        supported_languages: langs
      });
      this.importModal.classList.remove("open");
      this.loadModels();
    } catch (e) {
      alert(e.message);
    }
  }
}
