package com.google.mediapipe.examples.handlandmarker

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit

class SmartGlassesStreamService(private val listener: StreamListener) {

    interface StreamListener {
        fun onFrameReceived(bitmap: Bitmap)
        fun onConnectionStatusChanged(isConnected: Boolean, message: String? = null)
    }

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // Keep-alive
        .build()

    // Channel to handle backpressure (drop old frames)
    private val frameChannel = Channel<ByteArray>(Channel.CONFLATED)
    private val scope = CoroutineScope(Dispatchers.Default)

    init {
        startFrameProcessor()
    }

    fun connect(url: String) {
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "Connected to $url")
                listener.onConnectionStatusChanged(true, "Connected")
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                // Non-blocking send to channel (drops oldest if buffer full)
                frameChannel.trySend(bytes.toByteArray())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Handle text messages if any
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "Closing: $reason")
                listener.onConnectionStatusChanged(false, "Closing: $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Failure: ${t.message}", t)
                listener.onConnectionStatusChanged(false, "Error: ${t.message}")
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "User disconnected")
        webSocket = null
    }

    private fun startFrameProcessor() {
        scope.launch {
            for (bytes in frameChannel) {
                try {
                    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    if (bitmap != null) {
                        // Notify listener on Main thread? 
                        // Actually, let the listener handle threading or do it here if UI update is expected.
                        // However, inference should happen on background.
                        // We'll pass the raw bitmap and let the consumer decide.
                        listener.onFrameReceived(bitmap)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error decoding frame", e)
                }
            }
        }
    }

    companion object {
        private const val TAG = "SmartGlassesService"
    }
}
