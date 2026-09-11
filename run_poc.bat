@echo off
title AI Portable Language Translator - Local AI POC
echo =====================================================================
echo           AI PORTABLE LANGUAGE TRANSLATOR - LOCAL AI POC
echo =====================================================================
echo Strict Offline Mode: Zero Cloud STT / Zero Google Translate / Zero Cloud TTS
echo Starting local inference server on http://127.0.0.1:8000 ...
echo.

set PYTHON_CMD="C:\Users\HPS\AppData\Local\Programs\Python\Python312\python.exe"

if not exist %PYTHON_CMD% (
    set PYTHON_CMD=python
)

%PYTHON_CMD% -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
