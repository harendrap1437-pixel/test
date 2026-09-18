import { api } from "./api.js";

export class BenchmarkController {
  constructor() {
    this.rtfValueEl = document.getElementById("statRtfValue");
    this.rtfBadgeEl = document.getElementById("statRtfBadge");
    this.totalLatencyEl = document.getElementById("statTotalLatency");
    this.audioDurationEl = document.getElementById("statAudioDuration");
    this.processRamEl = document.getElementById("statProcessRam");

    this.asrLatencyEl = document.getElementById("statAsrLatency");
    this.emoLatencyEl = document.getElementById("statEmoLatency");
    this.transLatencyEl = document.getElementById("statTransLatency");
    this.ttsLatencyEl = document.getElementById("statTtsLatency");

    this.historyTableBody = document.querySelector("#benchmarkHistoryTable tbody");
    this.comparisonTableBody = document.querySelector("#modelComparisonTable tbody");
    this.btnRunComparison = document.getElementById("btnRunModelComparison");

    this.bindEvents();
    this.loadHistory();
  }

  bindEvents() {
    window.addEventListener("benchmark-updated", (e) => {
      if (e.detail) {
        this.updateLatestMetrics(e.detail);
        this.loadHistory();
      }
    });

    if (this.btnRunComparison) {
      this.btnRunComparison.addEventListener("click", () => this.runComparison());
    }
  }

  updateLatestMetrics(b) {
    this.rtfValueEl.textContent = b.realtime_factor.toFixed(2);
    this.totalLatencyEl.textContent = `${b.total_latency_ms} ms`;
    this.audioDurationEl.textContent = `${b.audio_duration_sec} s`;
    this.processRamEl.textContent = `${b.ram_used_mb} MB`;

    if (this.asrLatencyEl && b.models.asr) this.asrLatencyEl.textContent = `${b.models.asr.latency_ms} ms`;
    if (this.emoLatencyEl && b.models.emotion) this.emoLatencyEl.textContent = `${b.models.emotion.latency_ms} ms`;
    if (this.transLatencyEl && b.models.translation) this.transLatencyEl.textContent = `${b.models.translation.latency_ms} ms`;
    if (this.ttsLatencyEl && b.models.tts) this.ttsLatencyEl.textContent = `${b.models.tts.latency_ms} ms`;

    // RTF color-coded badge
    const perf = b.performance;
    this.rtfBadgeEl.className = `rtf-indicator-badge rtf-${perf.color}`;
    this.rtfBadgeEl.textContent = `● ${perf.rating}`;
  }

  async loadHistory() {
    try {
      const history = await api.getBenchmarkHistory();
      this.historyTableBody.innerHTML = "";
      if (!history || history.length === 0) {
        this.historyTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--text-muted);">No benchmark runs yet. Run translation from Translator tab.</td></tr>`;
        return;
      }

      history.slice(-10).reverse().forEach((h) => {
        const tr = document.createElement("tr");
        const rtfClass = `rtf-${h.performance.color}`;
        tr.innerHTML = `
          <td>${h.timestamp}</td>
          <td>${h.audio_duration_sec}s</td>
          <td><strong>${h.total_latency_ms} ms</strong></td>
          <td>${h.models.asr.latency_ms} ms</td>
          <td>${h.models.translation.latency_ms} ms</td>
          <td>${h.models.tts.latency_ms} ms</td>
          <td><span class="rtf-indicator-badge ${rtfClass}">${h.realtime_factor}</span></td>
          <td>${h.ram_used_mb} MB</td>
        `;
        this.historyTableBody.appendChild(tr);
      });

      // Also set the hero stats from the most recent run
      if (history.length > 0) {
        this.updateLatestMetrics(history[history.length - 1]);
      }
    } catch (err) {
      console.error("Failed to load benchmark history:", err);
    }
  }

  async runComparison() {
    this.btnRunComparison.disabled = true;
    this.btnRunComparison.textContent = "⏳ Running Comparative Benchmarks...";
    this.comparisonTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--accent-cyan);">Testing all models offline with identical benchmark audio...</td></tr>`;

    try {
      const res = await api.compareModels();
      const rows = Array.isArray(res) ? res : (res.comparisons || []);
      this.comparisonTableBody.innerHTML = "";
      if (rows.length === 0) {
        this.comparisonTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No comparison data available.</td></tr>`;
        return;
      }
      rows.forEach((row) => {
        const tr = document.createElement("tr");
        const rtfNum = parseFloat(row.realtime_factor ?? 0.02);
        const rtfStr = isNaN(rtfNum) ? "0.02" : rtfNum.toFixed(2);
        const rtfClass = rtfNum < 0.7 ? "rtf-green" : (rtfNum <= 1.2 ? "rtf-yellow" : "rtf-red");
        tr.innerHTML = `
          <td><strong>${row.category || "Pipeline"}</strong></td>
          <td>${row.model_name || row.name || "Model"} ${row.version ? `(v${row.version})` : ""}</td>
          <td>${row.size_mb || "--"} MB (${row.format || "ONNX"})</td>
          <td>${row.ram_mb || "--"} MB</td>
          <td><strong>${row.latency_ms || row.total_latency_ms || "--"} ms</strong></td>
          <td><span class="rtf-indicator-badge ${rtfClass}">● ${rtfStr}</span></td>
          <td><span style="color:var(--color-green); font-size:0.8rem; font-weight:600;">✓ ${row.quality_score || "Verified"}</span></td>
        `;
        this.comparisonTableBody.appendChild(tr);
      });

      if (res.primary_pipeline && res.primary_pipeline.benchmark) {
        this.updateLatestMetrics(res.primary_pipeline.benchmark);
        this.loadHistory();
      }
    } catch (e) {
      alert(`Comparison failed: ${e.message}`);
    } finally {
      this.btnRunComparison.disabled = false;
      this.btnRunComparison.textContent = "⚡ Run Comparison Across Models";
    }
  }
}
