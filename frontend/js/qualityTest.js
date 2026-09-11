import { api } from "./api.js";

export class QualityTestController {
  constructor() {
    this.tableBody = document.querySelector("#qualityTestTable tbody");
    this.scoreValEl = document.getElementById("qualityScoreVal");
    this.correctCountEl = document.getElementById("countCorrect");
    this.mostlyCountEl = document.getElementById("countMostly");
    this.needsCountEl = document.getElementById("countNeeds");
    this.btnRunAll = document.getElementById("btnRunAllQualityTests");

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
    this.tableBody.innerHTML = "";
    items.forEach((item) => {
      const tr = document.createElement("tr");

      tr.innerHTML = `
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
    const total = items.length;
    const correct = items.filter((d) => d.manual_evaluation === "Correct").length;
    const mostly = items.filter((d) => d.manual_evaluation === "Mostly correct").length;
    const needs = items.filter((d) => d.manual_evaluation === "Needs improvement").length;

    const scorePct = total > 0 ? (((correct * 1.0 + mostly * 0.75) / total) * 100).toFixed(1) : "0.0";

    this.scoreValEl.textContent = `${scorePct}%`;
    this.correctCountEl.textContent = correct;
    this.mostlyCountEl.textContent = mostly;
    this.needsCountEl.textContent = needs;
  }

  async runAllTests() {
    this.btnRunAll.disabled = true;
    this.btnRunAll.textContent = "⏳ Translating All 10 Categories Locally...";

    try {
      const res = await api.runQualityTest();
      this.renderTable(res.dataset);
      this.scoreValEl.textContent = `${res.prototype_evaluation_score_pct}%`;
      this.correctCountEl.textContent = res.correct_count;
      this.mostlyCountEl.textContent = res.mostly_correct_count;
      this.needsCountEl.textContent = res.needs_improvement_count;
    } catch (e) {
      alert(`Quality test error: ${e.message}`);
    } finally {
      this.btnRunAll.disabled = false;
      this.btnRunAll.textContent = "⚡ Run All 10 Tests Offline";
    }
  }
}
