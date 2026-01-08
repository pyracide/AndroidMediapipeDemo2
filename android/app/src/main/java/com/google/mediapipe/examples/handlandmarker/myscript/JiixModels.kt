package com.google.mediapipe.examples.handlandmarker.myscript

import com.google.gson.annotations.SerializedName

data class RecognitionRoot(
    val type: String?,
    val elements: List<Element>?,
    val version: String?
)

data class Element(
    val id: String,
    val type: String,
    @SerializedName("bounding-box")
    val boundingBox: BoundingBox,
    val words: List<Word>?,
    val label: String?
)

data class Word(
    val label: String?,
    @SerializedName("bounding-box")
    val boundingBox: BoundingBox?
)

data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)
