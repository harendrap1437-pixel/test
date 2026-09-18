@echo off
title Push AI Translator to GitHub
echo =====================================================================
echo               PUSHING AI TRANSLATOR TO GITHUB
echo =====================================================================
echo Target: https://github.com/harendrap1437-pixel/test.git
echo.
git push -u origin main
echo.
if %ERRORLEVEL% equ 0 (
    echo =====================================================================
    echo SUCCESS: Your project has been completely uploaded to GitHub!
    echo =====================================================================
) else (
    echo.
    echo If GitHub asks for login, please sign in via browser or enter your token.
)
pause
