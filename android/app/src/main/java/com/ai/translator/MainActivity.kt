package com.ai.translator

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.webkit.*
import android.widget.FrameLayout
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private val TAG = "MainActivity"
    private lateinit var webView: WebView
    private lateinit var pipelineEngine: OfflinePipelineEngine
    private var localServer: LocalAppServer? = null
    private var textToSpeech: TextToSpeech? = null
    private var isTtsReady = false
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private val PERMISSION_REQUEST_AUDIO = 101
    private var filePathCallback: ValueCallback<Array<Uri>>? = null

    private val fileChooserLauncher: ActivityResultLauncher<Intent> = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (filePathCallback != null) {
            val uris = WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
            filePathCallback?.onReceiveValue(uris)
            filePathCallback = null
        }
    }

    // High-reliability Google Speech Dialog Launcher fallback
    private val speechDialogLauncher: ActivityResultLauncher<Intent> = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        isListening = false
        if (result.resultCode == RESULT_OK && result.data != null) {
            val matches = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val recognizedText = matches?.firstOrNull() ?: ""
            if (recognizedText.isNotBlank()) {
                sendSpeechResultToWeb(recognizedText, isFinal = true)
            } else {
                sendSpeechErrorToWeb("No speech detected.")
            }
        } else {
            sendSpeechStateToWeb("idle")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Dark immersive status bar & navigation bar
        window.statusBarColor = Color.parseColor("#0a0d14")
        window.navigationBarColor = Color.parseColor("#0a0d14")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_VISIBLE
        }

        // Initialize Native Engines
        pipelineEngine = OfflinePipelineEngine(applicationContext)
        textToSpeech = try {
            TextToSpeech(this, this, "com.google.android.tts")
        } catch (e: Exception) {
            TextToSpeech(this, this)
        }

        // Start Local App Server on loopback
        localServer = LocalAppServer(applicationContext, pipelineEngine, 8765)
        val activePort = localServer?.start() ?: 8765

        // Create Fullscreen WebView
        val rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#0a0d14"))
        }

        webView = WebView(this).apply {
            setBackgroundColor(Color.parseColor("#0a0d14"))
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }
        rootLayout.addView(webView)
        setContentView(rootLayout)

        configureWebView()
        checkAudioPermissions()
        initSpeechRecognizer()

        // Load Local App URL
        webView.loadUrl("http://127.0.0.1:$activePort/index.html")

        // Handle shared audio if opened via intent
        intent?.let { handleIncomingAudioIntent(it) }

        // Back button navigation
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    finish()
                }
            }
        })
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        intent?.let { handleIncomingAudioIntent(it) }
    }

    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            allowFileAccess = true
            allowContentAccess = true
            mediaPlaybackRequiresUserGesture = false
            cacheMode = WebSettings.LOAD_DEFAULT
            useWideViewPort = true
            loadWithOverviewMode = true
            builtInZoomControls = false
            displayZoomControls = false
            setSupportZoom(false)
        }

        // Attach JavaScript Bridge
        webView.addJavascriptInterface(AndroidBridge(this, textToSpeech, pipelineEngine), "AndroidBridge")

        // Handle Audio Recording Permission & File Picking in WebView
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.let {
                    runOnUiThread {
                        it.grant(it.resources)
                    }
                }
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@MainActivity.filePathCallback?.onReceiveValue(null)
                this@MainActivity.filePathCallback = filePathCallback

                val intent = try {
                    fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                        type = "audio/*"
                        addCategory(Intent.CATEGORY_OPENABLE)
                    }
                } catch (e: Exception) {
                    Intent(Intent.ACTION_GET_CONTENT).apply {
                        type = "*/*"
                        addCategory(Intent.CATEGORY_OPENABLE)
                    }
                }

                try {
                    fileChooserLauncher.launch(intent)
                    return true
                } catch (e: Exception) {
                    this@MainActivity.filePathCallback?.onReceiveValue(null)
                    this@MainActivity.filePathCallback = null
                    return false
                }
            }
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                // Inject native ready signal & deliver any pending shared audio
                view?.evaluateJavascript(
                    """
                    if (window.AndroidBridge) { console.log('Connected to ' + window.AndroidBridge.getPlatformName()); }
                    if (window._pendingSharedAudio && typeof window.onSharedAudioReceived === 'function') {
                        var p = window._pendingSharedAudio;
                        window._pendingSharedAudio = null;
                        window.onSharedAudioReceived(p.name, p.dataUrl);
                    }
                    """.trimIndent(),
                    null
                )
            }
        }
    }

    private fun checkAudioPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_AUDIO
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_AUDIO) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initSpeechRecognizer()
                webView.reload()
            }
        }
    }

    private fun initSpeechRecognizer() {
        mainHandler.post {
            if (SpeechRecognizer.isRecognitionAvailable(this)) {
                try {
                    speechRecognizer?.destroy()
                    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
                        setRecognitionListener(object : RecognitionListener {
                            override fun onReadyForSpeech(params: Bundle?) {
                                isListening = true
                                sendSpeechStateToWeb("ready")
                            }

                            override fun onBeginningOfSpeech() {
                                sendSpeechStateToWeb("listening")
                            }

                            override fun onRmsChanged(rmsdB: Float) {}

                            override fun onBufferReceived(buffer: ByteArray?) {}

                            override fun onEndOfSpeech() {
                                isListening = false
                                sendSpeechStateToWeb("processing")
                            }

                            override fun onError(error: Int) {
                                isListening = false
                                Log.w(TAG, "SpeechRecognizer error: $error")
                                if (error == SpeechRecognizer.ERROR_CLIENT || error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY) {
                                    // Fallback to modal dialog intent if in-app recognizer has client issue
                                    launchSpeechDialog("hi-IN")
                                } else if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                                    sendSpeechErrorToWeb("No speech detected. Please speak clearly into the microphone.")
                                } else {
                                    sendSpeechErrorToWeb("Speech recognition issue ($error). Try speaking again.")
                                }
                            }

                            override fun onResults(results: Bundle?) {
                                isListening = false
                                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                val bestText = matches?.firstOrNull() ?: ""
                                if (bestText.isNotBlank()) {
                                    sendSpeechResultToWeb(bestText, isFinal = true)
                                } else {
                                    sendSpeechErrorToWeb("No clear words detected.")
                                }
                            }

                            override fun onPartialResults(partialResults: Bundle?) {
                                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                val partialText = matches?.firstOrNull() ?: ""
                                if (partialText.isNotBlank()) {
                                    sendSpeechResultToWeb(partialText, isFinal = false)
                                }
                            }

                            override fun onEvent(eventType: Int, params: Bundle?) {}
                        })
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to create SpeechRecognizer", e)
                }
            }
        }
    }

    fun startSpeech(langCode: String) {
        mainHandler.post {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                checkAudioPermissions()
                sendSpeechErrorToWeb("Microphone permission required.")
                return@post
            }

            val bcp47 = when (langCode.lowercase()) {
                "hi" -> "hi-IN"
                "bn" -> "bn-IN"
                "ta" -> "ta-IN"
                "te" -> "te-IN"
                "mr" -> "mr-IN"
                "gu" -> "gu-IN"
                "kn" -> "kn-IN"
                "ml" -> "ml-IN"
                "pa" -> "pa-IN"
                "ur" -> "ur-IN"
                else -> "en-US"
            }

            if (SpeechRecognizer.isRecognitionAvailable(this)) {
                if (speechRecognizer == null) {
                    initSpeechRecognizer()
                }
                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, bcp47)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, bcp47)
                    putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, bcp47)
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                    putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
                }
                try {
                    speechRecognizer?.cancel()
                    speechRecognizer?.startListening(intent)
                    isListening = true
                    sendSpeechStateToWeb("listening")
                } catch (e: Exception) {
                    Log.w(TAG, "SpeechRecognizer.startListening failed, fallback to dialog", e)
                    launchSpeechDialog(bcp47)
                }
            } else {
                launchSpeechDialog(bcp47)
            }
        }
    }

    fun stopSpeech() {
        mainHandler.post {
            try {
                if (isListening) {
                    speechRecognizer?.stopListening()
                }
            } catch (e: Exception) {
                // ignore
            }
        }
    }

    fun launchSpeechDialog(langTag: String) {
        mainHandler.post {
            try {
                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, langTag)
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak now in $langTag...")
                }
                speechDialogLauncher.launch(intent)
            } catch (e: Exception) {
                sendSpeechErrorToWeb("Speech recognition unavailable on device.")
            }
        }
    }

    fun sendSpeechResultToWeb(text: String, isFinal: Boolean) {
        runOnUiThread {
            val escaped = JSONObject.quote(text)
            val js = if (isFinal) {
                "window.onNativeSpeechResult && window.onNativeSpeechResult($escaped);"
            } else {
                "window.onNativeSpeechPartial && window.onNativeSpeechPartial($escaped);"
            }
            webView.evaluateJavascript(js, null)
        }
    }

    fun sendSpeechStateToWeb(state: String) {
        runOnUiThread {
            webView.evaluateJavascript("window.onNativeSpeechState && window.onNativeSpeechState('$state');", null)
        }
    }

    fun sendSpeechErrorToWeb(errorMsg: String) {
        runOnUiThread {
            val escaped = JSONObject.quote(errorMsg)
            webView.evaluateJavascript("window.onNativeSpeechError && window.onNativeSpeechError($escaped);", null)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech?.language = Locale.US
            textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    runOnUiThread {
                        webView.evaluateJavascript("window.onNativeTtsStart && window.onNativeTtsStart();", null)
                    }
                }

                override fun onDone(utteranceId: String?) {
                    runOnUiThread {
                        webView.evaluateJavascript("window.onNativeTtsDone && window.onNativeTtsDone();", null)
                    }
                }

                override fun onError(utteranceId: String?) {
                    runOnUiThread {
                        webView.evaluateJavascript("window.onNativeTtsDone && window.onNativeTtsDone();", null)
                    }
                }
            })
            isTtsReady = true
        }
    }

    private fun handleIncomingAudioIntent(intent: Intent) {
        val action = intent.action
        if (action != Intent.ACTION_SEND && action != Intent.ACTION_VIEW) return

        val uri: Uri? = if (action == Intent.ACTION_SEND) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(Intent.EXTRA_STREAM)
            } ?: intent.data
        } else {
            intent.data
        }

        if (uri == null) return

        Thread {
            try {
                var fileName = "shared_audio"
                val projection = arrayOf(OpenableColumns.DISPLAY_NAME)
                try {
                    contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
                        if (cursor.moveToFirst()) {
                            val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                            if (nameIdx >= 0) {
                                fileName = cursor.getString(nameIdx) ?: fileName
                            }
                        }
                    }
                } catch (e: Exception) {
                    fileName = uri.lastPathSegment ?: "shared_audio"
                }

                val mime = contentResolver.getType(uri) ?: "audio/wav"
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes != null && bytes.isNotEmpty()) {
                    val b64 = android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
                    val dataUrl = "data:$mime;base64,$b64"

                    mainHandler.post {
                        deliverSharedAudio(fileName, dataUrl)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error reading shared audio intent", e)
            }
        }.start()
    }

    private fun deliverSharedAudio(fileName: String, dataUrl: String) {
        val escapedName = JSONObject.quote(fileName)
        val escapedData = JSONObject.quote(dataUrl)
        val js = """
            (function() {
                if (typeof window.onSharedAudioReceived === 'function') {
                    window.onSharedAudioReceived($escapedName, $escapedData);
                } else {
                    window._pendingSharedAudio = { name: $escapedName, dataUrl: $escapedData };
                }
            })();
        """.trimIndent()
        webView.evaluateJavascript(js, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        localServer?.stop()
        try {
            speechRecognizer?.destroy()
        } catch (e: Exception) {}
        speechRecognizer = null
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        webView.destroy()
    }
}
