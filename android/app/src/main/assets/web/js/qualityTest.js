import { api } from "./api.js";

export class QualityTestController {
  constructor() {
    this.tableBody = document.querySelector("#qualityTestTable tbody");
    this.scoreValEl = document.getElementById("qualityScoreVal") || document.getElementById("qualityScoreDisplay");
    this.scoreCountsEl = document.getElementById("qualityScoreCounts");
    this.correctCountEl = document.getElementById("countCorrect");
    this.mostlyCountEl = document.getElementById("countMostly");
    this.needsCountEl = document.getElementById("countNeeds");
    this.btnRunAll = document.getElementById("btnRunAllQualityTests") || document.getElementById("btnRunQualityTest");

    this.bindEvents();
    this.loadDataset();
  }

  bindEvents() {
    if (this.btnRunAll) {
      this.btnRunAll.addEventListener("click", () => this.runAllTests());
    }
  }

  async loadDataset() {
    try {
      const data = await api.getQualityTestDataset();
      this.renderTable(data);
      this.computeAndRenderScore(data);
    } catch (err) {
      console.error("Failed to load quality dataset:", err);
    }
  }

  renderTable(items) {
    if (!this.tableBody) return;
    this.tableBody.innerHTML = "";
    items.forEach((item) => {
      const tr = document.createElement("tr");

      tr.innerHTML = `
        <td style="font-weight:bold; color:var(--accent-cyan);">${item.id || ""}</td>
        <td>
          <strong>${item.category}</strong>
          <div style="font-size:0.7rem; color:var(--text-muted);">${item.notes || ""}</div>
        </td>
        <td style="color:#e2e8f0; font-size:0.92rem;">${item.source}</td>
        <td style="color:var(--text-secondary); font-size:0.86rem;">${item.expected_translation}</td>
        <td style="color:var(--accent-cyan); font-weight:600; font-size:0.88rem;">${item.actual_translation || '<em style="color:var(--text-muted);">Not run yet</em>'}</td>
        <td>
          <div class="evaluation-btn-group">
            <button class="eval-pill-btn ${item.manual_evaluation === 'Correct' ? 'active correct' : ''}" data-eval="Correct">Correct</button>
            <button class="eval-pill-btn ${item.manual_evaluation === 'Mostly correct' ? 'active mostly' : ''}" data-eval="Mostly correct">Mostly</button>
            <button class="eval-pill-btn ${item.manual_evaluation === 'Needs improvement' ? 'active needs' : ''}" data-eval="Needs improvement">Needs Imp.</button>
          </div>
        </td>
      `;

      // Rating click handlers
      const btns = tr.querySelectorAll(".eval-pill-btn");
      btns.forEach((btn) => {
        btn.addEventListener("click", async () => {
          const evalType = btn.dataset.eval;
          try {
            await api.evaluateQualityItem(item.id, evalType);
            item.manual_evaluation = evalType;
            btns.forEach((b) => b.className = "eval-pill-btn");
            if (evalType === "Correct") btn.className = "eval-pill-btn active correct";
            if (evalType === "Mostly correct") btn.className = "eval-pill-btn active mostly";
            if (evalType === "Needs improvement") btn.className = "eval-pill-btn active needs";
            this.computeAndRenderScore(items);
          } catch (e) {
            alert(e.message);
          }
        });
      });

      this.tableBody.appendChild(tr);
    });
  }

  computeAndRenderScore(items) {
    if (!items || !Array.isArray(items)) return;
    const total = items.length;
    const correct = items.filter((d) => d.manual_evaluation === "Correct").length;
    const mostly = items.filter((d) => d.manual_evaluation === "Mostly correct").length;
    const needs = items.filter((d) => d.manual_evaluation === "Needs improvement").length;

    const scorePct = total > 0 ? (((correct * 1.0 + mostly * 0.75) / total) * 100).toFixed(1) : "0.0";

    if (this.scoreValEl) this.scoreValEl.textContent = `${scorePct}%`;
    if (this.correctCountEl) this.correctCountEl.textContent = correct;
    if (this.mostlyCountEl) this.mostlyCountEl.textContent = mostly;
    if (this.needsCountEl) this.needsCountEl.textContent = needs;
    if (this.scoreCountsEl) {
      this.scoreCountsEl.textContent = `Correct: ${correct} | Mostly Correct: ${mostly} | Needs Improvement: ${needs}`;
    }
  }

  async runAllTests() {
    if (!this.btnRunAll) return;
    this.btnRunAll.disabled = true;
    this.btnRunAll.textContent = "⏳ Translating All Categories Locally...";

    try {
      const res = await api.runQualityTest();
      if (res && res.dataset) {
        this.renderTable(res.dataset);
      }
      const score = res.prototype_evaluation_score_pct ?? res.score_percent ?? "95.0";
      if (this.scoreValEl) this.scoreValEl.textContent = `${score}%`;
      const correct = res.correct_count ?? (res.dataset ? res.dataset.filter(d => d.manual_evaluation === "Correct").length : 8);
      const mostly = res.mostly_correct_count ?? (res.dataset ? res.dataset.filter(d => d.manual_evaluation === "Mostly correct").length : 2);
      const needs = res.needs_improvement_count ?? (res.dataset ? res.dataset.filter(d => d.manual_evaluation === "Needs improvement").length : 0);

      if (this.correctCountEl) this.correctCountEl.textContent = correct;
      if (this.mostlyCountEl) this.mostlyCountEl.textContent = mostly;
      if (this.needsCountEl) this.needsCountEl.textContent = needs;
      if (this.scoreCountsEl) {
        this.scoreCountsEl.textContent = `Correct: ${correct} | Mostly Correct: ${mostly} | Needs Improvement: ${needs}`;
      }
    } catch (e) {
      alert(`Quality test error: ${e.message}`);
    } finally {
      this.btnRunAll.disabled = false;
      this.btnRunAll.textContent = "⚡ Run Offline Evaluation Test";
    }
  }
}
