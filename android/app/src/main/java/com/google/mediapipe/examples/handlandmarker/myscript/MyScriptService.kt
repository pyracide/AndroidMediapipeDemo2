package com.google.mediapipe.examples.handlandmarker.myscript

import android.content.Context
import android.util.DisplayMetrics
import android.util.Log
import com.google.gson.Gson
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

class MyScriptService(private val context: Context, private val listener: RecognitionListener) {

    interface RecognitionListener {
        fun onTextRecognized(text: String)
    }

    private var engine: Engine? = null
    private var offscreenEditor: OffscreenEditor? = null
    private var contentPart: ContentPart? = null
    private var converter: DisplayMetricsConverter? = null
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    
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
                            performExport()
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
                            
                            // 2. If null, create new. If default file locked/exists, use temp file.
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

    fun clear() {
        scope.launch(Dispatchers.Main) {
            if (enableEditorLogging) Log.d("Editor Logging", "Clearing editor")
            offscreenEditor?.clear()
        }
    }
    
    fun close() {
        offscreenEditor?.close()
        contentPart?.close()
    }

    private fun performExport() {
        scope.launch(Dispatchers.Default) {
            try {
                // FIXED: Pass emptyArray() instead of null
                val jiixString = offscreenEditor?.export_(emptyArray(), MimeType.JIIX)
                if (jiixString != null) {
                    if (enableEditorLogging) {
                        Log.d("Editor Logging", "JIIX Export: $jiixString")
                    }
                    val root = Gson().fromJson(jiixString, RecognitionRoot::class.java)
                    val textBuilder = StringBuilder()
                    
                    root.elements?.forEach { element ->
                        if (element.label != null) {
                            textBuilder.append(element.label).append(" ")
                        }
                        element.words?.forEach { word ->
                            if (word.label != null) {
                                textBuilder.append(word.label).append(" ")
                            }
                        }
                    }
                    
                    val resultText = textBuilder.toString().trim()
                    withContext(Dispatchers.Main) {
                        listener.onTextRecognized(resultText)
                    }
                }
            } catch (e: Exception) {
                Log.e("Editor Logging", "Export failed", e)
            }
        }
    }
    
    data class PointData(val x: Float, val y: Float, val timestamp: Long)
    
    companion object {
        private const val TAG = "MyScriptService"
    }
}
