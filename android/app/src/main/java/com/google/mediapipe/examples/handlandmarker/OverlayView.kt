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
    var isTapGesturesEnabled: Boolean = true
    private var isTapTriggered = false
    var coordinateScale: Float = 1.0f
    var sendMode: Int = 0 // 0 = Third Person, 1 = First Person
    var isCenterCrop: Boolean = false
    
    // Tap Gesture Detection State
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

    fun clearTracking() {
        results = null
        isTapTriggered = false
        isWriting = false
        invalidate()
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
                    // Check straight fingers for First Person mode
                    val indexStraight = isFingerStraight(landmark[5], landmark[6], landmark[8])
                    val middleStraight = isFingerStraight(landmark[9], landmark[10], landmark[12])
                    val ringStraight = isFingerStraight(landmark[13], landmark[14], landmark[16])
                    val pinkyStraight = isFingerStraight(landmark[17], landmark[18], landmark[20])
                    
                    for (normalizedLandmark in landmark) {
                        canvas.drawPoint(
                            normalizedLandmark.x() * imageWidth * scaleFactor + offsetX,
                            normalizedLandmark.y() * imageHeight * scaleFactor + offsetY,
                            pointPaint
                        )
                    }

                    HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                        val startIdx = connection!!.start()
                        val endIdx = connection.end()
                        
                        // Determine if this connection belongs to a straight finger
                        val isStraightFinger = when {
                            startIdx in 5..8 && endIdx in 5..8 -> indexStraight
                            startIdx in 9..12 && endIdx in 9..12 -> middleStraight
                            startIdx in 13..16 && endIdx in 13..16 -> ringStraight
                            startIdx in 17..20 && endIdx in 17..20 -> pinkyStraight
                            else -> false
                        }
                        
                        val currentPaint = if (isStraightFinger) {
                            android.graphics.Paint(linePaint).apply { color = android.graphics.Color.GREEN }
                        } else {
                            linePaint
                        }

                        canvas.drawLine(
                            landmark[startIdx].x() * imageWidth * scaleFactor + offsetX,
                            landmark[startIdx].y() * imageHeight * scaleFactor + offsetY,
                            landmark[endIdx].x() * imageWidth * scaleFactor + offsetX,
                            landmark[endIdx].y() * imageHeight * scaleFactor + offsetY,
                            currentPaint
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
                // Use max to match center-crop behavior (fixes miniature scaled down issue)
                // Use min to match fit-center behavior (standard for phone camera/mjpeg)
                if (isCenterCrop) {
                    max(width * 1f / imageWidth, height * 1f / imageHeight)
                } else {
                    min(width * 1f / imageWidth, height * 1f / imageHeight)
                }
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
        val STOP_THRESHOLD_TAP = 0.26
        
        // --- Tap Gesture Detection (Double/Triple Pinch) ---
        // Use a separate release threshold so tap sensitivity can be tuned independently of pen-up.
        if (isTapGesturesEnabled) {
            if (ratio < START_THRESHOLD && !isTapTriggered) {
                isTapTriggered = true
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastTapTime > 400) {
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
                    tapHandler.postDelayed(tapRunnable, 400)
                }
            } else if (ratio > STOP_THRESHOLD_TAP) {
                isTapTriggered = false
            }
        }

        // --- Pinch Detection (The "Pen" with Hysteresis) ---
        var justStarted = false
        if (!isWriting && ratio < START_THRESHOLD) {
            isWriting = true
            justStarted = true
            Log.d("OverlayView", "DOWN")
            // Start new path
            currentPath = Path()
            val avgX = (indexTip.x() + thumbTip.x()) / 2f
            val avgY = (indexTip.y() + thumbTip.y()) / 2f
            
            val x = avgX * imageWidth * scaleFactor + offsetX
            val y = avgY * imageHeight * scaleFactor + offsetY
            currentPath?.moveTo(x, y)
            currentStrokePoints.clear()
            
            val scaledX = (avgX * imageWidth * scaleFactor) * coordinateScale + offsetX
            val scaledY = (avgY * imageHeight * scaleFactor) * coordinateScale + offsetY
            currentStrokePoints.add(MyScriptService.PointData(scaledX, scaledY, System.currentTimeMillis()))
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
        
        if (isWriting && !justStarted) {
            val avgX = (indexTip.x() + thumbTip.x()) / 2f
            val avgY = (indexTip.y() + thumbTip.y()) / 2f
            
            val x = avgX * imageWidth * scaleFactor + offsetX
            val y = avgY * imageHeight * scaleFactor + offsetY
            currentPath?.lineTo(x, y)
            
            val scaledX = (avgX * imageWidth * scaleFactor) * coordinateScale + offsetX
            val scaledY = (avgY * imageHeight * scaleFactor) * coordinateScale + offsetY
            currentStrokePoints.add(MyScriptService.PointData(scaledX, scaledY, System.currentTimeMillis()))
        }
        
        // --- The "Send/Clear" Gesture ---
        var isClearGesture = false
        
        if (sendMode == 0) {
            // Third Person (Current) Mode
            val middleExt = distance(middleTip, wrist) > distance(middleMCP, wrist) * 0.85f
            val ringExt = distance(ringTip, wrist) > distance(ringMCP, wrist) * 0.85f
            val pinkyExt = distance(pinkyTip, wrist) > distance(pinkyMCP, wrist) * 0.85f
            
            val fingersOpen = middleExt && ringExt && pinkyExt
            val thumbIndexApart = ratio > 0.95
            isClearGesture = fingersOpen && thumbIndexApart && !isWriting
        } else {
            // First Person Mode: 4 fingers straight + sum > 320%
            val indexSim = getFingerStraightness(indexMCP, landmarks[6], indexTip)
            val middleSim = getFingerStraightness(middleMCP, landmarks[10], middleTip)
            val ringSim = getFingerStraightness(ringMCP, landmarks[14], ringTip)
            val pinkySim = getFingerStraightness(pinkyMCP, landmarks[18], pinkyTip)
            
            val totalSim = indexSim + middleSim + ringSim + pinkySim
            
            // Per finger threshold: 0.75 (75%)
            // Total threshold: 3.2 (320%)
            val allStraight = indexSim > 0.75f && middleSim > 0.75f && ringSim > 0.75f && pinkySim > 0.75f
            isClearGesture = allStraight && totalSim > 3.2f && !isWriting
        }
        
        if (isClearGesture) {
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
    
    // Returns the 2D vector similarity (cosine) between MCP->PIP and PIP->Tip
    private fun getFingerStraightness(mcp: NormalizedLandmark, pip: NormalizedLandmark, tip: NormalizedLandmark): Float {
        val mcpX = mcp.x() * imageWidth * scaleFactor
        val mcpY = mcp.y() * imageHeight * scaleFactor
        val pipX = pip.x() * imageWidth * scaleFactor
        val pipY = pip.y() * imageHeight * scaleFactor
        val tipX = tip.x() * imageWidth * scaleFactor
        val tipY = tip.y() * imageHeight * scaleFactor
        
        val v1x = pipX - mcpX
        val v1y = pipY - mcpY
        val v2x = tipX - pipX
        val v2y = tipY - pipY
        
        val dotProduct = v1x * v2x + v1y * v2y
        val mag1 = sqrt((v1x * v1x + v1y * v1y).toDouble()).toFloat()
        val mag2 = sqrt((v2x * v2x + v2y * v2y).toDouble()).toFloat()
        
        if (mag1 == 0f || mag2 == 0f) return 0f
        return dotProduct / (mag1 * mag2)
    }

    // Checks if the 2D vector from knuckle to PIP is within a similarity threshold (cosine) to PIP to Tip
    private fun isFingerStraight(mcp: NormalizedLandmark, pip: NormalizedLandmark, tip: NormalizedLandmark, threshold: Float = 0.75f): Boolean {
        return getFingerStraightness(mcp, pip, tip) > threshold
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
    }
}
