package com.google.mediapipe.examples.handlandmarker

import android.media.MediaCodec
import android.media.MediaFormat
import android.view.Surface
import android.util.Log
import java.nio.ByteBuffer

class H264Decoder(private val surface: Surface) {

    private var decoder: MediaCodec? = null
    private var isDecoding = false
    private var outputThread: Thread? = null

    // Optional callback for when a frame is physically rendered to the surface
    var onFrameRenderedListener: (() -> Unit)? = null

    fun start() {
        try {
            decoder = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, 1280, 720) 
            // Width and height will be automatically updated by SPS/PPS config packets
            
            decoder?.configure(format, surface, null, 0)
            decoder?.start()
            isDecoding = true
            
            outputThread = Thread {
                processOutput()
            }
            outputThread?.start()
            Log.d(TAG, "H264 Hardware Decoder started.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start decoder", e)
        }
    }

    private fun getNALType(data: ByteArray): Int {
        var offset = 0
        while (offset < data.size - 4) { // Ensure safe bounds
            if (data[offset].toInt() == 0 && data[offset+1].toInt() == 0) {
                if (data[offset+2].toInt() == 1) {
                    return data[offset+3].toInt() and 0x1F
                } else if (data[offset+2].toInt() == 0 && data[offset+3].toInt() == 1) {
                    return data[offset+4].toInt() and 0x1F
                }
            }
            offset++
        }
        return -1
    }

    fun decodeNalUnit(nalUnit: ByteArray) {
        if (!isDecoding || decoder == null) return
        
        try {
            val inputBufferIndex = decoder!!.dequeueInputBuffer(10000)
            if (inputBufferIndex >= 0) {
                val inputBuffer = decoder!!.getInputBuffer(inputBufferIndex)
                inputBuffer?.clear()
                inputBuffer?.put(nalUnit)
                
                val type = getNALType(nalUnit)
                var flags = 0
                if (type == 7 || type == 8) { // 7 = SPS, 8 = PPS
                    flags = MediaCodec.BUFFER_FLAG_CODEC_CONFIG
                }
                
                decoder!!.queueInputBuffer(
                    inputBufferIndex,
                    0,
                    nalUnit.size,
                    System.nanoTime() / 1000,
                    flags
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error queuing input buffer", e)
        }
    }

    private fun processOutput() {
        val info = MediaCodec.BufferInfo()
        while (isDecoding) {
            try {
                val outputBufferIndex = decoder?.dequeueOutputBuffer(info, 10000) ?: -1
                if (outputBufferIndex >= 0) {
                    // True = render to Surface
                    decoder?.releaseOutputBuffer(outputBufferIndex, true)
                    
                    // Notify that a frame has been pushed to the GPU
                    onFrameRenderedListener?.invoke()
                } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val newFormat = decoder?.outputFormat
                    Log.d(TAG, "Decoder output format changed: $newFormat")
                }
            } catch (e: Exception) {
                if (isDecoding) {
                    Log.e(TAG, "Error processing output buffer", e)
                }
                if (e is IllegalStateException) {
                    break
                }
            }
        }
    }

    fun stop() {
        isDecoding = false
        try {
            outputThread?.join(500)
            decoder?.stop()
            decoder?.release()
            decoder = null
            Log.d(TAG, "H264 Hardware Decoder stopped.")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping decoder", e)
        }
    }

    companion object {
        private const val TAG = "H264Decoder"
    }
}
