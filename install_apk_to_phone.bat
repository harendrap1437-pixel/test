@echo off
title Install AI Portable Translator APK to Phone
echo ====================================================
echo Installing AI Portable Language Translator APK...
echo ====================================================
echo.

if exist "ai-portable-translator-final.apk" (
    set "APK_PATH=ai-portable-translator-final.apk"
    set "PKG_NAME=com.ai.translator"
) else if exist "ai-portable-translator-release.apk" (
    set "APK_PATH=ai-portable-translator-release.apk"
    set "PKG_NAME=com.ai.translator"
) else if exist "ai-portable-translator.apk" (
    set "APK_PATH=ai-portable-translator.apk"
    set "PKG_NAME=com.ai.translator.debug"
) else (
    set "APK_PATH=android\app\build\outputs\apk\release\app-release.apk"
    set "PKG_NAME=com.ai.translator"
)

echo Target APK: %APK_PATH%
echo Target Package: %PKG_NAME%
echo.
echo Checking for connected Android devices via ADB...
C:\adb\adb.exe devices

echo.
echo Installing APK to device...
C:\adb\adb.exe install -r -d "%APK_PATH%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================
    echo [SUCCESS] APK installed successfully!
    echo Launching app on device...
    echo ====================================================
    C:\adb\adb.exe shell am start -n %PKG_NAME%/com.ai.translator.MainActivity
) else (
    echo.
    echo ====================================================
    echo [NOTE] ADB installation failed or device offline.
    echo To install manually:
    echo 1. Copy "%APK_PATH%" to your phone storage.
    echo 2. Open Files app on phone and tap to install.
    echo ====================================================
)

pause
