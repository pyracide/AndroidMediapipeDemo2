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
    private val modelLock = Any()

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
            synchronized(modelLock) {
                if (!isLoaded) {
                    FloatArray(candidates.size) { Float.NEGATIVE_INFINITY }
                } else {
                    rankCandidatesNative(prompt, candidates.toTypedArray(), true) // Using sequence scoring by default
                }
            }
        }
    }

    fun getLastTokenCounts(): IntArray {
        return getLastTokenCountsNative()
    }

    private var modelFileName = MODEL_FILE_Q4

    fun setModelFileName(newName: String) {
        if (modelFileName == newName) return
        synchronized(modelLock) {
            modelFileName = newName
            release()
        }
    }

    private fun loadModelInternal(): Boolean {
        val modelFile = File(context.filesDir, modelFileName)
        if (!modelFile.exists()) {
            val copied = copyAsset(modelFileName, modelFile)
            if (!copied) {
                Log.e(TAG, "Failed to copy model asset: $modelFileName")
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
    private external fun rankCandidatesNative(prompt: String, candidates: Array<String>, useSequenceScoring: Boolean): FloatArray
    private external fun getLastTokenCountsNative(): IntArray
    private external fun resetNativeContext()
    private external fun cancelNativeInference()
    private external fun freeModel()

    companion object {
        private const val TAG = "LlmEngine"
        private const val MODEL_FILE_Q4 = "SmolLM2-360M.Q4_0.gguf"
        private const val MODEL_FILE_Q8 = "SmolLM2-360M.Q8_0.gguf"
        private const val NUM_THREADS = 4

        fun modelFileNameForIndex(index: Int): String {
            return if (index == 1) MODEL_FILE_Q8 else MODEL_FILE_Q4
        }

        init {
            System.loadLibrary("llm")
        }
    }
}
