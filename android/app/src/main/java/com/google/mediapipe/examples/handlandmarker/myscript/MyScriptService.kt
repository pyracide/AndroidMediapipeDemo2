package com.google.mediapipe.examples.handlandmarker.myscript

import android.content.Context
import android.util.DisplayMetrics
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.google.mediapipe.examples.handlandmarker.MyScriptApplication
import com.myscript.iink.ContentPart
import com.myscript.iink.EditorError
import com.myscript.iink.Engine
import com.myscript.iink.IOffscreenEditorListener
import com.myscript.iink.MimeType
import com.myscript.iink.OffscreenEditor
import com.myscript.iink.PointerEvent
import com.myscript.iink.PointerEventType
import com.myscript.iink.PointerType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.text.Normalizer
import java.util.concurrent.atomic.AtomicBoolean

// --- Data Models (Moved here to ensure compilation) ---
data class RecognitionRoot(
    val type: String?,
    val elements: List<Element>?,
    val version: String?
)

data class Element(
    val id: String,
    val type: String,
    @SerializedName("bounding-box")
    val boundingBox: BoundingBox?,
    val words: List<Word>?,
    val items: List<Item>?, 
    val label: String?
)

data class Word(
    val label: String?,
    val candidates: List<String>?,
    @SerializedName("bounding-box")
    val boundingBox: BoundingBox?,
    val items: List<Item>? 
)

data class Item(
    val type: String,
    val id: String?,
    @SerializedName("X")
    val X: List<Float>?,
    @SerializedName("Y")
    val Y: List<Float>?
)

data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)
// ----------------------------------------------------

class MyScriptService(private val context: Context, private val listener: RecognitionListener) {

    enum class DecoderMode {
        LLM,
        NGRAM,
        NONE
    }

    interface RecognitionListener {
        fun onTextRecognized(text: String, debugText: String = "")
        fun onJiixReceived(items: List<Item>)
        fun onLlmStatus(status: String) {}
    }

    private var engine: Engine? = null
    private var offscreenEditor: OffscreenEditor? = null
    private var contentPart: ContentPart? = null
    private var converter: DisplayMetricsConverter? = null
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    private val languageModel by lazy { TrigramLanguageModel(context) }
    private val llmEngine by lazy { LlmEngine(context) }
    private val llmContextWords = mutableListOf<String>()
    private var llmLoaded = false
    private var canUndo = false
    
    private val enableEditorLogging = true
    private var isReady = false
    private val isInitializing = AtomicBoolean(false)

    init {
        val app = context.applicationContext as? MyScriptApplication
        engine = app?.engine
        
        if (engine == null) {
            Log.e("Editor Logging", "Engine is null in MyScriptService. Check Application initialization logs.")
        } else {
            initializeEditor()
        }
    }

    fun setDisplayMetrics(metrics: DisplayMetrics) {
        converter = DisplayMetricsConverter(metrics)
        Log.d("Editor Logging", "DisplayMetrics set. Converter initialized: ${converter != null}")
    }
    
    fun setNgWeight(weight: Float) {
        languageModel.ngWeight = weight
    }
    
    fun setDebugMode(enabled: Boolean) {
        isDebugEnabled = enabled
        languageModel.isDebugMode = enabled
    }
    
    var decoderMode = DecoderMode.LLM
    var oneWordOnly = true
    private var isDebugEnabled = false
    var llmTimeoutMs: Long = 1000L

    fun resetLlmContext() {
        llmContextWords.clear()
        canUndo = false
        llmEngine.resetContext()
    }

    fun preloadLlm() {
        scope.launch {
            listener.onLlmStatus("LLM: loading")
            val loaded = llmEngine.ensureLoaded()
            llmLoaded = loaded
            listener.onLlmStatus(if (loaded) "LLM: loaded" else "LLM: load failed")
        }
    }

    fun setLlmModelIndex(index: Int) {
        llmLoaded = false
        llmEngine.setModelFileName(LlmEngine.modelFileNameForIndex(index))
    }

    fun undoLastWord(): Boolean {
        if (!canUndo || llmContextWords.isEmpty()) return false
        llmContextWords.removeAt(llmContextWords.size - 1)
        canUndo = false
        return true
    }

    private fun initializeEditor() {
        if (isInitializing.getAndSet(true)) {
            Log.d("Editor Logging", "Initialization already in progress, skipping.")
            return
        }

        try {
            // Create Editor immediately on Main Thread
            if (offscreenEditor == null && engine != null) {
                // Use standard screen size
                val width = context.resources.displayMetrics.widthPixels.toFloat().coerceAtLeast(100f)
                val height = context.resources.displayMetrics.heightPixels.toFloat().coerceAtLeast(100f)
                
                offscreenEditor = engine?.createOffscreenEditor(width, height)
                
                if (offscreenEditor == null) {
                    Log.e("Editor Logging", "Failed to create OffscreenEditor (returned null)")
                    isInitializing.set(false)
                    return
                }
                
                offscreenEditor?.addListener(object : IOffscreenEditorListener {
                    override fun partChanged(editor: OffscreenEditor) {
                        if (enableEditorLogging) Log.d("Editor Logging", "partChanged")
                    }

                    override fun contentChanged(editor: OffscreenEditor, blockIds: Array<out String>) {
                            if (enableEditorLogging) Log.d("Editor Logging", "contentChanged: ${blockIds.joinToString()}")
                            // NO-OP: We wait for explicit commitAndClear()
                    }

                    override fun onError(editor: OffscreenEditor, blockId: String, err: EditorError, message: String) {
                        Log.e("Editor Logging", "onError: blockId=$blockId, error=${err.name}, message=$message")
                    }
                })
                Log.d("Editor Logging", "OffscreenEditor created")
            }

            // Background init
            scope.launch(Dispatchers.IO) {
                try {
                    val partConf = try {
                        context.assets.open("part_conf.json").bufferedReader().use { it.readText() }
                    } catch (e: Exception) {
                        Log.e("Editor Logging", "Failed to load part_conf.json", e)
                        null
                    }

                    val dataDir = File(context.filesDir, "myscript_data")
                    if (!dataDir.exists()) dataDir.mkdirs()
                    // Default file
                    val partFile = File(dataDir, "content.iink")
                    
                    withContext(Dispatchers.Main) {
                        // Inject configuration
                        if (partConf != null && offscreenEditor?.configuration != null) {
                            try {
                                offscreenEditor?.configuration?.inject(partConf)
                                offscreenEditor?.configuration?.setBoolean("export.jiix.text.words", true)
                                if (enableEditorLogging) Log.d("Editor Logging", "Configuration injected")
                            } catch (e: Exception) {
                                Log.e("Editor Logging", "Failed to inject configuration", e)
                            }
                        }

                        // Create/Open Part with Auto-Recovery
                        try {
                            contentPart = null
                            
                            // 1. Try to open existing default file
                            if (partFile.exists()) {
                                try {
                                    contentPart = engine?.openPackage(partFile)?.getPart(0)
                                } catch (e: Exception) {
                                    Log.w("Editor Logging", "Failed to open existing package. Will try to create new.", e)
                                }
                            }
                            
                            // 2. If null, create new. If default file locked, use temp file.
                            if (contentPart == null) {
                                try {
                                    // Try deleting default file first
                                    if (partFile.exists() && !partFile.delete()) {
                                        Log.w("Editor Logging", "Could not delete locked content.iink. Creating temporary file.")
                                        // Use a unique file if delete failed
                                        val tempFile = File(dataDir, "content_${System.currentTimeMillis()}.iink")
                                        contentPart = engine?.createPackage(tempFile)?.createPart("raw-content")
                                    } else {
                                        // File deleted or didn't exist, create fresh
                                        contentPart = engine?.createPackage(partFile)?.createPart("raw-content")
                                    }
                                } catch (e: Exception) {
                                    Log.e("Editor Logging", "createPart('raw-content') failed", e)
                                    // Retry with Title Case if lowercase failed (rare but possible mismatch)
                                    try {
                                        // Use another unique file to be safe
                                        val retryFile = File(dataDir, "retry_${System.currentTimeMillis()}.iink")
                                        contentPart = engine?.createPackage(retryFile)?.createPart("Raw Content")
                                    } catch (e2: Exception) {
                                        Log.e("Editor Logging", "createPart('Raw Content') also failed", e2)
                                        throw e2 // Give up
                                    }
                                }
                            }
                            
                            if (contentPart != null) {
                                offscreenEditor?.part = contentPart
                                isReady = true
                                Log.d("Editor Logging", "MyScript Editor Fully Initialized. Part: ${contentPart?.type}")
                            } else {
                                Log.e("Editor Logging", "Fatal: ContentPart is null after all attempts")
                            }
                        } catch (e: Exception) {
                            Log.e("Editor Logging", "Fatal exception in part creation", e)
                        } finally {
                            isInitializing.set(false)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("Editor Logging", "Background init failed", e)
                    isInitializing.set(false)
                }
            }

        } catch (e: Exception) {
            Log.e("Editor Logging", "Init failed", e)
            isInitializing.set(false)
        }
    }

    fun addStroke(points: List<PointData>) {
        if (!isReady) {
            Log.w("Editor Logging", "addStroke ignored: Editor not ready yet. Retrying initialization...")
            if (!isInitializing.get()) {
                initializeEditor()
            }
            return
        }
        if (converter == null) {
            Log.e("Editor Logging", "addStroke ignored: Converter is null")
            return
        }
        
        if (points.isEmpty()) return

        if (enableEditorLogging) {
            Log.d("Editor Logging", "addStroke: Processing ${points.size} points.")
        }
        
        scope.launch(Dispatchers.Main) {
            try {
                val pointerEvents = ArrayList<PointerEvent>()
                
                points.forEachIndexed { index, point ->
                    val xMm = converter!!.x_px2mm(point.x)
                    val yMm = converter!!.y_px2mm(point.y)
                    
                    if (index == 0) {
                        pointerEvents.add(PointerEvent(PointerEventType.DOWN, xMm, yMm, point.timestamp, 0f, PointerType.PEN, 0))
                    } else {
                        pointerEvents.add(PointerEvent(PointerEventType.MOVE, xMm, yMm, point.timestamp, 0f, PointerType.PEN, 0))
                    }
                }
                
                val lastPoint = points.last()
                val lastXMm = converter!!.x_px2mm(lastPoint.x)
                val lastYMm = converter!!.y_px2mm(lastPoint.y)
                pointerEvents.add(PointerEvent(PointerEventType.UP, lastXMm, lastYMm, lastPoint.timestamp, 0f, PointerType.PEN, 0))

                offscreenEditor?.addStrokes(pointerEvents.toTypedArray(), true)
            } catch (e: Exception) {
                Log.e("Editor Logging", "Error in addStroke", e)
            }
        }
    }

    fun commitAndClear() {
        scope.launch(Dispatchers.Default) {
            try {
                // Safety check
                if (offscreenEditor?.part == null) {
                    Log.e("Editor Logging", "Commit ignored: Editor has no part.")
                    return@launch
                }

                if (enableEditorLogging) Log.d("Editor Logging", "Commit triggered. Waiting for idle...")
                offscreenEditor?.waitForIdle()
                if (enableEditorLogging) Log.d("Editor Logging", "Engine idle. Exporting...")

                // 1. Export JIIX
                val jiixString = offscreenEditor?.export_(emptyArray(), MimeType.JIIX)
                
                if (jiixString != null) {
                    if (enableEditorLogging) {
                        Log.d("Editor Logging", "JIIX Export: ${jiixString.take(100)}...")
                    }
                    val root = Gson().fromJson(jiixString, RecognitionRoot::class.java)
                    val allItems = mutableListOf<Item>()
                    
                    val textSequence = mutableListOf<List<String>>()
                    val fallbackText = java.lang.StringBuilder()
                    
                    root.elements?.forEach { element ->
                        // Collect Text
                        if (!element.words.isNullOrEmpty()) {
                            element.words.forEach { word ->
                                val cands = mutableListOf<String>()
                                word.label?.let { cands.add(it) } // Add the default label first
                                word.candidates?.forEach { if (it != word.label) cands.add(it) }
                                
                                if (cands.isNotEmpty()) {
                                    textSequence.add(cands)
                                }
                                
                                if (word.label != null) {
                                    fallbackText.append(word.label).append(" ")
                                }
                                // Collect Strokes inside Words
                                word.items?.let { allItems.addAll(it) }
                            }
                        } else if (element.label != null) {
                            textSequence.add(listOf(element.label))
                            fallbackText.append(element.label).append(" ")
                        }
                        
                        // Collect Strokes inside Elements (Raw Content structure)
                        element.items?.let { allItems.addAll(it) }
                    }
                    
                    // Evaluate Best String Match 
                    var finalDebugText = ""
                    
                    val sequenceToEvaluate = if (oneWordOnly && textSequence.isNotEmpty()) {
                        listOf(textSequence.last())
                    } else {
                        textSequence
                    }
                    
                    val finalFallback = if (oneWordOnly) {
                        fallbackText.toString().trim().split(Regex("\\s+")).lastOrNull() ?: ""
                    } else {
                        fallbackText.toString().trim()
                    }

                    val resultText = when (decoderMode) {
                        DecoderMode.NGRAM -> {
                            if (sequenceToEvaluate.isNotEmpty()) {
                                val decodeResult = languageModel.decodeOptimalSentence(sequenceToEvaluate)
                                if (isDebugEnabled) {
                                    finalDebugText = decodeResult.debugInfo
                                }
                                decodeResult.text
                            } else {
                                finalFallback
                            }
                        }
                        DecoderMode.LLM -> {
                            val candidateList = sequenceToEvaluate.lastOrNull() ?: emptyList()
                            val llmCandidates = candidateList.map { normalizeLlmInput(it) }
                            if (candidateList.isEmpty()) {
                                finalFallback
                            } else {
                                val rawPrompt = llmContextWords.joinToString(" ")
                                val prompt = normalizeLlmInput(rawPrompt)
                                val startTime = System.currentTimeMillis()
                                var llmScores: FloatArray? = null
                                var llmTokenCounts: IntArray = IntArray(0)
                                var timedOut = false

                                if (!llmLoaded) {
                                    withContext(Dispatchers.Main) {
                                        listener.onLlmStatus("LLM: loading")
                                    }
                                }

                                try {
                                    llmScores = withTimeout(llmTimeoutMs) {
                                        llmEngine.rankCandidates(prompt, llmCandidates)
                                    }
                                    llmTokenCounts = llmEngine.getLastTokenCounts()
                                } catch (e: TimeoutCancellationException) {
                                    timedOut = true
                                    llmEngine.cancelInference()
                                }

                                if (!llmLoaded && llmScores != null && llmScores.isNotEmpty()) {
                                    llmLoaded = true
                                    withContext(Dispatchers.Main) {
                                        listener.onLlmStatus("LLM: loaded")
                                    }
                                } else if (!llmLoaded && (llmScores == null || llmScores.isEmpty())) {
                                    withContext(Dispatchers.Main) {
                                        listener.onLlmStatus("LLM: load failed")
                                    }
                                }

                                val inferenceMs = System.currentTimeMillis() - startTime

                                if (timedOut || llmScores == null || llmScores.isEmpty()) {
                                    if (isDebugEnabled) {
                                        finalDebugText = "LLM timeout (>1000 ms). Fallback to JIIX top.\n" +
                                            "Inference: ${inferenceMs} ms"
                                    }
                                    finalFallback
                                } else {
                                    val llmWeights = listOf(0.88f, 0.66f, 0.44f, 0.22f, 0.0f)
                                    val llmOrder = llmScores.indices.sortedByDescending { llmScores[it] }
                                    val llmRanks = IntArray(llmScores.size)
                                    llmOrder.forEachIndexed { rank, idx ->
                                        llmRanks[idx] = rank
                                    }

                                    var bestIndex = 0
                                    var bestScore = -1e9f

                                    val debugBuilder = StringBuilder()
                                    if (isDebugEnabled) {
                                        val rawContextLine = if (rawPrompt.isBlank()) "<empty>" else rawPrompt
                                        val contextLine = if (prompt.isBlank()) "<empty>" else prompt
                                        debugBuilder.append("**Context (raw):** ${rawContextLine}\n")
                                        debugBuilder.append("**Context (normalized):** ${contextLine}\n")
                                        debugBuilder.append("**Mode:** LLM\n")
                                        debugBuilder.append("**Inference:** ${inferenceMs} ms\n")
                                        debugBuilder.append("**JIIX candidates:** ${candidateList.take(5).joinToString(", ")}\n")
                                        val llmCandWithTok = llmCandidates.mapIndexed { index, cand ->
                                            val tok = llmTokenCounts.getOrElse(index) { 0 }
                                            if (tok > 0) "${cand} (${tok} tok)" else cand
                                        }
                                        debugBuilder.append("**LLM candidates:** ${llmCandWithTok.take(5).joinToString(", ")}\n")
                                        val rawPairs = llmCandidates.mapIndexed { index, cand ->
                                            val raw = llmScores.getOrElse(index) { -1e9f }
                                            "${cand}=${String.format("%.4f", raw)}"
                                        }
                                        debugBuilder.append("**LLM raw logits:** ${rawPairs.joinToString(", ")}\n")
                                    }

                                    candidateList.forEachIndexed { index, cand ->
                                        val jiixScore = languageModel.getJiixScore(index)
                                        val llmRank = llmRanks.getOrElse(index) { llmScores.size }
                                        val llmWeight = llmWeights.getOrElse(llmRank) { 0f }
                                        val total = jiixScore + llmWeight
                                        if (total > bestScore) {
                                            bestScore = total
                                            bestIndex = index
                                        }

                                        if (isDebugEnabled) {
                                            val llmScore = llmScores.getOrElse(index) { -1e9f }
                                            val tok = llmTokenCounts.getOrElse(index) { 0 }
                                            val tokLabel = if (tok > 0) " (${tok} tok)" else ""
                                            debugBuilder.append(
                                                String.format(
                                                    "%s%s (JIIX R%d=%.2f | LLM R%d=%.4f | W=%.2f | Tot=%.2f)\n",
                                                    cand,
                                                    tokLabel,
                                                    index + 1,
                                                    jiixScore,
                                                    llmRank + 1,
                                                    llmScore,
                                                    llmWeight,
                                                    total
                                                )
                                            )
                                        }
                                    }

                                    if (isDebugEnabled) {
                                        finalDebugText = debugBuilder.toString().trimEnd()
                                    }

                                    candidateList[bestIndex]
                                }
                            }
                        }
                        DecoderMode.NONE -> {
                            finalFallback
                        }
                    }
                    
                    if (isDebugEnabled) {
                        val fullJiixExport = "RAW JIIX (Pre-Trim):\n" + textSequence.mapIndexed { i, cands -> "W${i + 1}: " + cands.take(5).joinToString(", ") }.joinToString("\n")
                        if (finalDebugText.isNotBlank()) {
                            finalDebugText = fullJiixExport + "\n\n" + finalDebugText
                        } else {
                            finalDebugText = fullJiixExport
                        }
                    }
                    
                    if (enableEditorLogging) {
                        Log.d("Editor Logging", "Final Output: $resultText (vs raw: ${fallbackText.toString().trim()})")
                    }
                    
                    withContext(Dispatchers.Main) {
                        listener.onTextRecognized(resultText, finalDebugText)
                        
                        // Send strokes to visualizer
                        if (allItems.isNotEmpty()) {
                            listener.onJiixReceived(allItems)
                        }
                        
                        if (enableEditorLogging) Log.d("Editor Logging", "Result emitted. Clearing engine.")
                        offscreenEditor?.clear()
                    }

                    if (resultText.isNotBlank()) {
                        llmContextWords.add(resultText.trim())
                        canUndo = true
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        offscreenEditor?.clear()
                    }
                }
            } catch (e: Exception) {
                Log.e("Editor Logging", "Commit failed", e)
            }
        }
    }
    
    fun clear() {
        scope.launch(Dispatchers.Main) {
            if (enableEditorLogging) Log.d("Editor Logging", "Clearing editor")
            offscreenEditor?.clear()
        }
    }
    
    fun resetLanguageModelHistory() {
        languageModel.resetHistory()
        if (enableEditorLogging) Log.d("Editor Logging", "N-Gram language model history reset due to hand tracker loss.")
    }
    
    fun close() {
        llmEngine.release()
        offscreenEditor?.close()
        contentPart?.close()
    }
    
    data class PointData(val x: Float, val y: Float, val timestamp: Long)
    
    companion object {
        private const val TAG = "MyScriptService"
    }

    private fun normalizeLlmInput(value: String): String {
        if (value.isBlank()) return ""
        val nfkc = Normalizer.normalize(value, Normalizer.Form.NFKC)
        val noZeroWidth = nfkc
            .replace("\u200B", "")
            .replace("\u200C", "")
            .replace("\u200D", "")
            .replace("\uFEFF", "")
            .replace("\u00A0", " ")
        return noZeroWidth.replace(Regex("\\s+"), " ").trim()
    }
}
