package com.google.mediapipe.examples.handlandmarker.myscript

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class LlmEngine(private val context: Context) {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    @Volatile private var loadDeferred: CompletableDeferred<Boolean>? = null
    @Volatile private var isLoaded = false

    suspend fun ensureLoaded(): Boolean {
        if (isLoaded) return true
        val deferred: CompletableDeferred<Boolean>
        val isLoader: Boolean
        synchronized(this) {
            val existing = loadDeferred
            if (existing != null) {
                deferred = existing
                isLoader = false
            } else {
                val created = CompletableDeferred<Boolean>()
                loadDeferred = created
                deferred = created
                isLoader = true
            }
        }

        if (isLoader) {
            val success = withContext(scope.coroutineContext) { loadModelInternal() }
            deferred.complete(success)
        }

        val result = deferred.await()
        isLoaded = result
        return result
    }

    fun resetContext() {
        resetNativeContext()
    }

    fun cancelInference() {
        cancelNativeInference()
    }

    fun release() {
        isLoaded = false
        loadDeferred = null
        freeModel()
    }

    suspend fun rankCandidates(prompt: String, candidates: List<String>): FloatArray {
        if (candidates.isEmpty()) return FloatArray(0)
        val ready = ensureLoaded()
        if (!ready) return FloatArray(candidates.size) { Float.NEGATIVE_INFINITY }
        return withContext(Dispatchers.Default) {
            rankCandidatesNative(prompt, candidates.toTypedArray())
        }
    }

    private fun loadModelInternal(): Boolean {
        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        if (!modelFile.exists()) {
            val copied = copyAsset(MODEL_FILE_NAME, modelFile)
            if (!copied) {
                Log.e(TAG, "Failed to copy model asset: $MODEL_FILE_NAME")
                return false
            }
        }
        return loadModel(modelFile.absolutePath, NUM_THREADS)
    }

    private fun copyAsset(filename: String, destFile: File): Boolean {
        return try {
            context.assets.open(filename).use { inputStream ->
                FileOutputStream(destFile).use { outputStream ->
                    val buffer = ByteArray(1024 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error copying LLM asset", e)
            false
        }
    }

    private external fun loadModel(modelPath: String, numThreads: Int): Boolean
    private external fun rankCandidatesNative(prompt: String, candidates: Array<String>): FloatArray
    private external fun resetNativeContext()
    private external fun cancelNativeInference()
    private external fun freeModel()

    companion object {
        private const val TAG = "LlmEngine"
        private const val MODEL_FILE_NAME = "SmolLM2-360M.Q4_0.gguf"
        private const val NUM_THREADS = 4

        init {
            System.loadLibrary("llm")
        }
    }
}
