# ProGuard / R8 Rules for AI Portable Language Translator
# 100% Offline Android App — WebView + Embedded LocalAppServer Architecture

# ============================================================
# 1. Keep WebView JavaScript Interface (Critical)
# ============================================================
# AndroidBridge is exposed to JavaScript via @JavascriptInterface.
# R8 must NOT strip or rename any @JavascriptInterface-annotated methods.
-keepclassmembers class com.ai.translator.AndroidBridge {
    @android.webkit.JavascriptInterface <methods>;
}
-keepattributes JavascriptInterface

# ============================================================
# 2. Keep Application Components
# ============================================================
-keep class com.ai.translator.MainActivity { *; }
-keep class com.ai.translator.LocalAppServer { *; }
-keep class com.ai.translator.OfflinePipelineEngine { *; }
-keep class com.ai.translator.AndroidBridge { *; }
-keep class com.ai.translator.NativeAudioRecorder { *; }

# Sherpa ONNX JNI Bindings (Must keep all classes, methods and fields for native C++ JNI)
-keep class com.k2fsa.sherpa.onnx.** { *; }
-keepclassmembers class com.k2fsa.sherpa.onnx.** { *; }
-dontwarn com.k2fsa.sherpa.onnx.**

# ============================================================
# 3. Kotlin Metadata & Reflection
# ============================================================
-keepattributes *Annotation*
-keepattributes Signature
-keepattributes InnerClasses
-keepattributes EnclosingMethod
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**
-dontwarn kotlinx.**

# ============================================================
# 4. Android Framework
# ============================================================
-keep class * extends android.app.Activity
-keep class * extends android.app.Application
-keep class * extends android.app.Service
-keep class * extends android.content.BroadcastReceiver

# ============================================================
# 5. AndroidX / Material Components
# ============================================================
-dontwarn com.google.android.material.**
-dontwarn androidx.**
-keep class com.google.android.material.** { *; }
-keep class androidx.** { *; }
-keep interface androidx.** { *; }

# ============================================================
# 6. JSON Parsing (org.json — Android built-in)
# ============================================================
-keep class org.json.** { *; }

# ============================================================
# 7. WebView
# ============================================================
-keepclassmembers class * extends android.webkit.WebViewClient {
    public void *(android.webkit.WebView, java.lang.String, android.graphics.Bitmap);
    public boolean *(android.webkit.WebView, java.lang.String);
    public void *(android.webkit.WebView, java.lang.String);
}
-keepclassmembers class * extends android.webkit.WebChromeClient {
    public void *(android.webkit.WebView, java.lang.String);
}

# ============================================================
# 8. General Optimizations
# ============================================================
-optimizationpasses 5
-dontusemixedcaseclassnames
-verbose
-dontpreverify

# Remove logging in release builds
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int d(...);
    public static int i(...);
}
