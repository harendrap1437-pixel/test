@echo off
title Build AI Portable Translator APK
echo ====================================================
echo Compiling Standalone Offline Android APKs...
echo ====================================================
echo.

cd /d "%~dp0"
echo [Sync] Copying frontend assets to Android assets...
xcopy /s /e /y "frontend\*" "android\app\src\main\assets\web\"

cd /d "%~dp0android"
echo [1/2] Building Release APK...
call .\gradlew.bat assembleRelease

echo.
echo [2/2] Building Debug APK...
call .\gradlew.bat assembleDebug

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================
    echo [SUCCESS] APKs compiled and signed successfully!
    echo ====================================================
    cd /d "%~dp0"
    if not exist "dist" mkdir "dist"
    copy /y "android\app\build\outputs\apk\release\app-release.apk" "ai-portable-translator-release.apk"
    copy /y "android\app\build\outputs\apk\release\app-release.apk" "ai-portable-translator-final.apk"
    copy /y "android\app\build\outputs\apk\release\app-release.apk" "dist\ai-portable-translator-release.apk"
    copy /y "android\app\build\outputs\apk\release\app-release.apk" "dist\ai-portable-translator-final.apk"
    copy /y "android\app\build\outputs\apk\debug\app-debug.apk" "ai-portable-translator.apk"
    copy /y "android\app\build\outputs\apk\debug\app-debug.apk" "dist\ai-portable-translator.apk"
    echo.
    echo Saved binaries:
    echo   - Final:   %~dp0ai-portable-translator-final.apk
    echo   - Release: %~dp0ai-portable-translator-release.apk
    echo   - Debug:   %~dp0ai-portable-translator.apk
    echo   - Dist:    %~dp0dist\
    echo.
) else (
    echo.
    echo [ERROR] Build failed! Check Gradle output above.
)

pause
