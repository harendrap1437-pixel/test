import time
from playwright.sync_api import sync_playwright
from pathlib import Path

brain_dir = Path(r"C:\Users\HPS\.gemini\antigravity-ide\brain\51979550-90f5-4e67-8ef3-72c19f32a712")

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 850})
        page.goto("http://127.0.0.1:8000")
        page.wait_for_timeout(1000)

        # 1. Translator Tab with File Mode and Run Translate
        page.click("#modeFileBtn")
        page.wait_for_timeout(400)
        sample_btns = page.locator(".sample-audio-btn")
        if sample_btns.count() > 0:
            sample_btns.first.click()
            page.wait_for_timeout(300)
        page.click("#translateActionBtn")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(brain_dir / "screenshot_01_translator_active.png"))
        print("Captured screenshot_01_translator_active.png")

        # 2. Model Manager Tab
        page.click("button[data-tab='tabModelManager']")
        page.wait_for_timeout(800)
        page.screenshot(path=str(brain_dir / "screenshot_02_model_manager.png"))
        print("Captured screenshot_02_model_manager.png")

        # 3. Benchmark Tab & Model Comparison
        page.click("button[data-tab='tabBenchmark']")
        page.wait_for_timeout(500)
        page.click("#btnRunModelComparison")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(brain_dir / "screenshot_03_benchmark.png"))
        print("Captured screenshot_03_benchmark.png")

        # 4. Quality Evaluation Tab & Run 10 Tests
        page.click("button[data-tab='tabQuality']")
        page.wait_for_timeout(500)
        page.click("#btnRunAllQualityTests")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(brain_dir / "screenshot_04_quality_evaluation.png"))
        print("Captured screenshot_04_quality_evaluation.png")

        # 5. Airplane Mode Test Tab & Run Audit
        page.click("button[data-tab='tabAirplane']")
        page.wait_for_timeout(500)
        page.click("#btnRunOfflineAudit")
        page.wait_for_timeout(800)
        page.screenshot(path=str(brain_dir / "screenshot_05_airplane_mode.png"))
        print("Captured screenshot_05_airplane_mode.png")

        browser.close()
        print("All 5 verification screenshots successfully captured!")

if __name__ == "__main__":
    capture_all()
