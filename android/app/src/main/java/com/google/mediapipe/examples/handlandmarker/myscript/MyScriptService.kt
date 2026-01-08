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

class MyScriptService(private val context: Context, private val listener: RecognitionListener) {

    interface RecognitionListener {
        fun onTextRecognized(text: String)
    }

    private var engine: Engine? = null
    private var offscreenEditor: OffscreenEditor? = null
    private var contentPart: ContentPart? = null
    private var converter: DisplayMetricsConverter? = null
    private val scope = CoroutineScope(Dispatchers.Main + Job())

    init {
        val app = context.applicationContext as? MyScriptApplication
        engine = app?.engine
        
        if (engine == null) {
            Log.e(TAG, "Engine is null. Is MyScriptApplication registered in AndroidManifest?")
        } else {
            initializeEditor()
        }
    }

    fun setDisplayMetrics(metrics: DisplayMetrics) {
        converter = DisplayMetricsConverter(metrics)
    }

    private fun initializeEditor() {
        scope.launch(Dispatchers.IO) {
            try {
                // Read configuration from assets
                val partConf = try {
                    context.assets.open("part_conf.json").bufferedReader().use { it.readText() }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load part_conf.json", e)
                    null
                }

                withContext(Dispatchers.Main) {
                    offscreenEditor = engine?.createOffscreenEditor(1f, 1f)
                    
                    // Inject the configuration
                    if (partConf != null) {
                        offscreenEditor?.configuration?.inject(partConf)
                    } else {
                        Log.w(TAG, "Proceeding without part_conf.json injection")
                    }

                    offscreenEditor?.addListener(object : IOffscreenEditorListener {
                        override fun partChanged(editor: OffscreenEditor?) {}

                        override fun contentChanged(editor: OffscreenEditor?, blockIds: Array<out String>?) {
                             performExport()
                        }

                        override fun onError(editor: OffscreenEditor?, blockId: String?, err: EditorError?, message: String?) {
                            Log.e(TAG, "Editor Error: $message")
                        }
                    })

                    // Create a Part
                    val dataDir = File(context.filesDir, "myscript_data")
                    dataDir.mkdirs()
                    val partFile = File(dataDir, "content.iink")
                    
                    contentPart = if (partFile.exists()) {
                         engine?.openPackage(partFile)?.getPart(0)
                    } else {
                         // Use "Raw Content" as requested by user
                         engine?.createPackage(partFile)?.createPart("Raw Content")
                    }
                    
                    offscreenEditor?.part = contentPart
                    Log.d(TAG, "MyScript Editor Initialized")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to init editor", e)
            }
        }
    }

    fun addStroke(points: List<PointData>) {
        if (offscreenEditor == null || converter == null) return
        
        scope.launch(Dispatchers.Main) {
            val pointerEvents = points.mapIndexed { index, point ->
                val eventType = when (index) {
                    0 -> PointerEventType.DOWN
                    points.size - 1 -> PointerEventType.UP
                    else -> PointerEventType.MOVE
                }
                
                // Convert pixels to mm
                val xMm = converter!!.x_px2mm(point.x)
                val yMm = converter!!.y_px2mm(point.y)
                
                PointerEvent(eventType, xMm, yMm, point.timestamp, 0f, PointerType.PEN, 0)
            }.toTypedArray()

            offscreenEditor?.addStrokes(pointerEvents, true)
        }
    }

    fun clear() {
        scope.launch(Dispatchers.Main) {
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
                // Export JIIX
                val jiixString = offscreenEditor?.export_(null, MimeType.JIIX)
                if (jiixString != null) {
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
                Log.e(TAG, "Export failed", e)
            }
        }
    }
    
    data class PointData(val x: Float, val y: Float, val timestamp: Long)
    
    companion object {
        private const val TAG = "MyScriptService"
    }
}
