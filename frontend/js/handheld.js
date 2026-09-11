/**
 * Handheld Hardware Device Simulator Controller
 */
export class HandheldDevice {
  constructor() {
    this.canvas = document.getElementById("audioWaveCanvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.ledGreen = document.getElementById("ledGreen");
    this.ledBlue = document.getElementById("ledBlue");
    this.ledAmber = document.getElementById("ledAmber");
    this.pttBtn = document.getElementById("pttMainBtn");
    
    this.isVisualizing = false;
    this.animationId = null;
    this.wavePhase = 0;

    this.initHardwareLeds();
    this.resizeCanvas();
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.parentElement.clientWidth || 300;
    this.canvas.height = 60;
  }

  initHardwareLeds() {
    // Green LED on by default for 100% Offline Mode
    this.setLed("green", true);
    this.setLed("blue", false);
    this.setLed("amber", false);
  }

  setLed(color, isOn) {
    const el = document.getElementById(`led${color.charAt(0).toUpperCase() + color.slice(1)}`);
    if (el) {
      if (isOn) el.classList.add("on");
      else el.classList.remove("on");
    }
  }

  startWaveform(active = true) {
    this.isVisualizing = true;
    this.setLed("blue", true);
    const draw = () => {
      if (!this.isVisualizing || !this.ctx) return;
      const w = this.canvas.width;
      const h = this.canvas.height;
      this.ctx.clearRect(0, 0, w, h);

      this.ctx.beginPath();
      this.ctx.lineWidth = 2.5;
      this.ctx.strokeStyle = active ? "#00f2fe" : "#4facfe";

      const sliceWidth = w / 40;
      let x = 0;
      this.wavePhase += 0.08;

      for (let i = 0; i < 40; i++) {
        const amplitude = active ? (h / 2.8) * Math.sin(this.wavePhase + i * 0.3) : 3;
        const y = h / 2 + amplitude;
        if (i === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
        x += sliceWidth;
      }
      this.ctx.stroke();
      this.animationId = requestAnimationFrame(draw);
    };
    draw();
  }

  stopWaveform() {
    this.isVisualizing = false;
    if (this.animationId) cancelAnimationFrame(this.animationId);
    this.setLed("blue", false);
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      // Draw flat line
      this.ctx.beginPath();
      this.ctx.lineWidth = 1;
      this.ctx.strokeStyle = "#1a273b";
      this.ctx.moveTo(0, this.canvas.height / 2);
      this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
      this.ctx.stroke();
    }
  }
}
