# =====================================================================
#  AI PORTABLE LANGUAGE TRANSLATOR - GOOGLE COLAB ONE-SHOT RUNNER
# =====================================================================
# Paste this entire block into a single Google Colab cell and press PLAY.

import os, sys, time, subprocess, re

print("🚀 [1/5] Setting up Project Directory...")
# Check if Google Drive is mounted and folder exists
drive_path = "/content/drive/MyDrive/ai speec reco"
if os.path.exists(drive_path):
    print(f"📁 Found Google Drive folder! Using: {drive_path}")
    os.chdir(drive_path)
else:
    print("🌐 Cloning latest repository from GitHub...")
    os.chdir("/content")
    if not os.path.exists("/content/test"):
        subprocess.run(["git", "clone", "https://github.com/harendrap1437-pixel/test.git"], check=True)
    os.chdir("/content/test")

print("📦 [2/5] Installing dependencies (fastapi, uvicorn, onnxruntime, etc.)...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi>=0.110.0", "uvicorn>=0.28.0", "python-multipart", "pydantic",
    "psutil", "soundfile", "numpy", "scipy", "onnxruntime>=1.16.0",
    "ctranslate2>=4.0.0", "transformers>=4.38.0", "sentencepiece", "sherpa_onnx>=1.10.0"
], check=True)

print("⚡ [3/5] Installing Cloudflare Tunnel (Linux)...")
subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/usr/local/bin/cloudflared"], check=True)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

print("🤖 [4/5] Starting AI Application Backend on port 8000...")
server_process = subprocess.Popen([
    sys.executable, "-m", "uvicorn", "backend.app:app",
    "--host", "0.0.0.0", "--port", "8000"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)

print("🌐 [5/5] Generating Public HTTPS Live URL...")
tunnel_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stderr=subprocess.PIPE, text=True
)

url_found = False
for line in iter(tunnel_proc.stderr.readline, ''):
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        live_url = match.group(0)
        print("\n" + "="*70)
        print("  🎉 YOUR AI TRANSLATOR IS LIVE ON THE CLOUD!")
        print(f"  👉 PUBLIC URL: {live_url}")
        print("="*70 + "\n")
        url_found = True
        break

if not url_found:
    print("Cloudflare started. Check logs above if URL didn't parse.")

# Keep cell alive so Colab doesn't exit
tunnel_proc.wait()
