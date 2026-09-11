@echo off
title AI Portable Language Translator - Online Public Tunnel
echo =====================================================================
echo          AI PORTABLE LANGUAGE TRANSLATOR - PUBLIC TUNNEL
echo =====================================================================
echo Starting Cloudflare Tunnel to expose local server to the internet...
echo.
if not exist "cloudflared.exe" (
    echo Downloading cloudflared...
    curl.exe -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)

cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
