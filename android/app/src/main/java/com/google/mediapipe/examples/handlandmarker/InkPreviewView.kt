package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.examples.handlandmarker.myscript.Item
import kotlin.math.max
import kotlin.math.min

class InkPreviewView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val paint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }

    private val strokes = mutableListOf<Path>()

    fun setStrokes(items: List<Item>) {
        strokes.clear()
        if (items.isEmpty()) {
            invalidate()
            return
        }

        // 1. Calculate Bounds
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        // Fix: Use -MAX_VALUE to ensure we find max correctly (MIN_VALUE is closest to 0 positive)
        var maxX = -Float.MAX_VALUE 
        var maxY = -Float.MAX_VALUE
        var hasPoints = false

        items.forEach { item ->
            if (item.type == "stroke" && item.X != null && item.Y != null) {
                for (i in item.X.indices) {
                    val x = item.X[i]
                    val y = item.Y[i]
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                    hasPoints = true
                }
            }
        }

        if (!hasPoints) {
            invalidate()
            return
        }

        // 2. Calculate Scale to fit View
        val padding = 20f
        val contentWidth = maxX - minX
        val contentHeight = maxY - minY
        
        val safeWidth = if (contentWidth > 0) contentWidth else 1f
        val safeHeight = if (contentHeight > 0) contentHeight else 1f

        val viewWidth = width.toFloat() - (padding * 2)
        val viewHeight = height.toFloat() - (padding * 2)

        val scaleX = viewWidth / safeWidth
        val scaleY = viewHeight / safeHeight
        val scale = min(scaleX, scaleY)

        // Center offsets
        val offsetX = padding + (viewWidth - (contentWidth * scale)) / 2f
        val offsetY = padding + (viewHeight - (contentHeight * scale)) / 2f

        // 3. Create Paths
        items.forEach { item ->
            if (item.type == "stroke" && item.X != null && item.Y != null) {
                val path = Path()
                for (i in item.X.indices) {
                    val x = (item.X[i] - minX) * scale + offsetX
                    val y = (item.Y[i] - minY) * scale + offsetY
                    if (i == 0) {
                        path.moveTo(x, y)
                    } else {
                        path.lineTo(x, y)
                    }
                }
                strokes.add(path)
            }
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK) // Background
        strokes.forEach { path ->
            canvas.drawPath(path, paint)
        }
    }
}
