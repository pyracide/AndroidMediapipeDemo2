/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.examples.handlandmarker.myscript.MyScriptService
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: HandLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()
    private var drawingPaint = Paint()
    private var drawingDotPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var offsetX: Float = 0f
    private var offsetY: Float = 0f

    // Drawing Mode State
    var isDrawingMode: Boolean = false
        set(value) {
            field = value
            if (!value) {
                clearDrawing()
            }
            invalidate()
        }
    private var isWriting: Boolean = false
    private val drawnPaths = mutableListOf<Path>()
    private var currentPath: Path? = null
    
    // MyScript Integration
    private val currentStrokePoints = mutableListOf<MyScriptService.PointData>()
    var strokeListener: OnStrokeListener? = null
    private var lastClearTime = 0L
    private var tapCount = 0
    private var lastTapTime = 0L
    private val tapHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val tapRunnable = Runnable {
        if (tapCount == 2) {
            strokeListener?.onDoublePinch()
        }
        tapCount = 0
    }
    var isTapGesturesEnabled = true
    private var isTapTriggered = false

    interface OnStrokeListener {
        fun onStroke(points: List<MyScriptService.PointData>)
        fun onClear()
        fun onDoublePinch()
        fun onTriplePinch()
        fun onDebugCoords(x: Float, y: Float)
    }

    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        clearDrawing()
        invalidate()
        initPaints()
    }
    
    fun clearDrawing() {
        drawnPaths.clear()
        currentPath = null
        isWriting = false
        currentStrokePoints.clear()
        Log.d("OverlayView", "CLEAR")
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
        
        drawingPaint.color = Color.GREEN
        drawingPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        drawingPaint.style = Paint.Style.STROKE
        drawingPaint.strokeJoin = Paint.Join.ROUND
        drawingPaint.strokeCap = Paint.Cap.ROUND
        
        drawingDotPaint.color = Color.GREEN
        drawingDotPaint.strokeWidth = LANDMARK_STROKE_WIDTH * 2
        drawingDotPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        
        if (isDrawingMode) {
            // Draw all saved paths
            for (path in drawnPaths) {
                canvas.drawPath(path, drawingPaint)
            }
            // Draw current path
            currentPath?.let {
                canvas.drawPath(it, drawingPaint)
            }
            
            // Draw dot if writing
            if (isWriting && results?.landmarks()?.isNotEmpty() == true) {
                 val landmark = results!!.landmarks().first()
                 val thumbTip = landmark[4]
                 val indexTip = landmark[8]
                 
                 val avgX = (indexTip.x() + thumbTip.x()) / 2f
                 val avgY = (indexTip.y() + thumbTip.y()) / 2f
                 
                 canvas.drawPoint(
                     avgX * imageWidth * scaleFactor + offsetX,
                     avgY * imageHeight * scaleFactor + offsetY,
                     drawingDotPaint
                 )
            }
            
        } else {
            // Normal Skeleton Mode
            results?.let { handLandmarkerResult ->
                for (landmark in handLandmarkerResult.landmarks()) {
                    for (normalizedLandmark in landmark) {
                        canvas.drawPoint(
                            normalizedLandmark.x() * imageWidth * scaleFactor + offsetX,
                            normalizedLandmark.y() * imageHeight * scaleFactor + offsetY,
                            pointPaint
                        )
                    }

                    HandLandmarker.HAND_CONNECTIONS.forEach {
                        canvas.drawLine(
                            landmark.get(it!!.start())
                                .x() * imageWidth * scaleFactor + offsetX,
                            landmark.get(it.start())
                                .y() * imageHeight * scaleFactor + offsetY,
                            landmark.get(it.end())
                                .x() * imageWidth * scaleFactor + offsetX,
                            landmark.get(it.end())
                                .y() * imageHeight * scaleFactor + offsetY,
                            linePaint
                        )
                    }
                }
            }
        }
    }

    fun setResults(
        handLandmarkerResults: HandLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = handLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in fitCenter mode.
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        
        offsetX = (width - imageWidth * scaleFactor) / 2f
        offsetY = (height - imageHeight * scaleFactor) / 2f

        if (handLandmarkerResults.landmarks().isNotEmpty()) {
            val firstHand = handLandmarkerResults.landmarks().first()
            if (isDrawingMode) {
                processGesture(firstHand)
            } else {
                // Just track midpoint for debug
                val thumbTip = firstHand[4]
                val indexTip = firstHand[8]
                val avgX = (indexTip.x() + thumbTip.x()) / 2f
                val avgY = (indexTip.y() + thumbTip.y()) / 2f
                
                val x = avgX * imageWidth * scaleFactor + offsetX
                val y = avgY * imageHeight * scaleFactor + offsetY
                strokeListener?.onDebugCoords(x, y)
            }
        }

        invalidate()
    }
    
    private fun processGesture(landmarks: List<NormalizedLandmark>) {
        val wrist = landmarks[0]
        val thumbTip = landmarks[4]
        val indexMCP = landmarks[5]
        val indexTip = landmarks[8]
        val middleMCP = landmarks[9]
        val middleTip = landmarks[12]
        val ringMCP = landmarks[13]
        val ringTip = landmarks[16]
        val pinkyMCP = landmarks[17]
        val pinkyTip = landmarks[20]
        
        val scaleDist = distance(wrist, indexMCP)
        val pinchDist = distance(thumbTip, indexTip)
        val ratio = if (scaleDist > 0) pinchDist / scaleDist else 100f // prevent div by zero
        
        // --- Pinch Detection (The "Pen") ---
        val START_THRESHOLD = 0.20
        val STOP_THRESHOLD = 0.30
        
        // --- Tap Gesture Detection (Double/Triple Pinch) ---
        // We use an independent trigger without hysteresis for "snappy" detection
        if (isTapGesturesEnabled) {
            if (ratio < START_THRESHOLD && !isTapTriggered) {
                isTapTriggered = true
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastTapTime > 500) {
                    tapCount = 1
                } else {
                    tapCount++
                }
                lastTapTime = currentTime
                
                tapHandler.removeCallbacks(tapRunnable)
                if (tapCount == 3) {
                    Log.d("OverlayView", "TRIPLE PINCH")
                    strokeListener?.onTriplePinch()
                    tapCount = 0
                } else {
                    tapHandler.postDelayed(tapRunnable, 500)
                }
            } else if (ratio > START_THRESHOLD) {
                isTapTriggered = false
            }
        }

        // --- Pinch Detection (The "Pen" with Hysteresis) ---
        if (!isWriting && ratio < START_THRESHOLD) {
            isWriting = true
            Log.d("OverlayView", "DOWN")
            // Start new path
            currentPath = Path()
            val avgX = (indexTip.x() + thumbTip.x()) / 2f
            val avgY = (indexTip.y() + thumbTip.y()) / 2f
            
            val x = avgX * imageWidth * scaleFactor + offsetX
            val y = avgY * imageHeight * scaleFactor + offsetY
            currentPath?.moveTo(x, y)
            currentStrokePoints.clear()
            currentStrokePoints.add(MyScriptService.PointData(x, y, System.currentTimeMillis()))
        } else if (isWriting && ratio > STOP_THRESHOLD) {
            isWriting = false
            Log.d("OverlayView", "UP")
            // Commit path
            currentPath?.let { drawnPaths.add(it) }
            currentPath = null
            
            // Notify listener
            if (currentStrokePoints.isNotEmpty()) {
                strokeListener?.onStroke(ArrayList(currentStrokePoints))
                currentStrokePoints.clear()
            }
        }
        
        if (isWriting) {
            val avgX = (indexTip.x() + thumbTip.x()) / 2f
            val avgY = (indexTip.y() + thumbTip.y()) / 2f
            
            val x = avgX * imageWidth * scaleFactor + offsetX
            val y = avgY * imageHeight * scaleFactor + offsetY
            currentPath?.lineTo(x, y)
            currentStrokePoints.add(MyScriptService.PointData(x, y, System.currentTimeMillis()))
        }
        
        // --- The "Clear" Gesture ---
        // Fingers: Middle (12), Ring (16), Pinky (20)
        // Condition: Tip significantly further from Wrist than Base (MCP)
        val middleExt = distance(middleTip, wrist) > distance(middleMCP, wrist) * 0.85
        val ringExt = distance(ringTip, wrist) > distance(ringMCP, wrist) * 0.85
        val pinkyExt = distance(pinkyTip, wrist) > distance(pinkyMCP, wrist) * 0.85
        
        val fingersOpen = middleExt && ringExt && pinkyExt
        val thumbIndexApart = ratio > 0.95
        
        if (fingersOpen && thumbIndexApart) {
             val currentTime = System.currentTimeMillis()
             // Debounce clear to avoid multiple triggers
             if (currentTime - lastClearTime > 1000) {
                 clearDrawing()
                 strokeListener?.onClear()
                 lastClearTime = currentTime
             }
        }
        
        // Also emit debug coords when drawing mode is ON
        val indexX = indexTip.x() * imageWidth * scaleFactor + offsetX
        val indexY = indexTip.y() * imageHeight * scaleFactor + offsetY
        strokeListener?.onDebugCoords(indexX, indexY)
    }
    
    private fun distance(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        // We use normalized coordinates directly? 
        // No, the prompt specifies "Euclidean distance". 
        // If we use normalized, X and Y have different scales unless aspect ratio is 1:1.
        // It's safer to use pixel coordinates to represent actual physical distance on screen.
        
        val x1 = p1.x() * imageWidth * scaleFactor
        val y1 = p1.y() * imageHeight * scaleFactor
        val x2 = p2.x() * imageWidth * scaleFactor
        val y2 = p2.y() * imageHeight * scaleFactor
        
        return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
    }
}
