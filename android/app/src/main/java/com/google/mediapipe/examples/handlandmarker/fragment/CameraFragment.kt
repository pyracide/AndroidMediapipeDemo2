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
package com.google.mediapipe.examples.handlandmarker.fragment

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.EditText
import android.widget.Toast
import androidx.camera.core.CameraControl
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.mediapipe.examples.handlandmarker.HandLandmarkerHelper
import com.google.mediapipe.examples.handlandmarker.MainViewModel
import com.google.mediapipe.examples.handlandmarker.OverlayView
import com.google.mediapipe.examples.handlandmarker.R
import com.google.mediapipe.examples.handlandmarker.SmartGlassesStreamService
import com.google.mediapipe.examples.handlandmarker.H264Decoder
import com.google.mediapipe.examples.handlandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.examples.handlandmarker.databinding.InfoBottomSheetBinding
import com.google.mediapipe.examples.handlandmarker.myscript.Item
import com.google.mediapipe.examples.handlandmarker.myscript.MyScriptService
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import androidx.camera.camera2.interop.Camera2CameraInfo
import android.hardware.camera2.CameraCharacteristics
import android.os.Handler
import android.os.Looper
import com.alexvas.rtsp.widget.RtspSurfaceView
import com.alexvas.rtsp.widget.RtspStatusListener

class CameraFragment : Fragment(), HandLandmarkerHelper.LandmarkerListener, TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "CameraFragment"
        private const val MODE_CLASSIC = 0
        private const val MODE_H264_CLASSIC = 1
        private const val MODE_H264_RTSP = 2
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var handLandmarkerHelper: HandLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT
    private var isWideAngle = false
    private var isDrawingMode = false
    private var cameraControl: CameraControl? = null
    
    // Blink state to interrupt MediaPipe feed
    @Volatile
    private var isBlinking = false
    
    // Smart Glasses Mode
    private var isSmartGlassesMode = false
    private var isSmartGlassesFlipped = false
    private var isSmartGlassesMirrored = false
    private var smartGlassesService: SmartGlassesStreamService? = null
    private var lastSocketUrl = "ws://192.168.1.218:81"
    private var h264Decoder: H264Decoder? = null
    
    // RTSP Player
    private var currentStreamMode = MODE_CLASSIC
    
    // Polling for RtspSurfaceView
    private var processingRunnable: Runnable? = null
    private val processingHandler = Handler(Looper.getMainLooper())


    
    private var myScriptService: MyScriptService? = null
    private var tts: TextToSpeech? = null
    private var settingsDialog: BottomSheetDialog? = null
    private var bottomSheetBinding: InfoBottomSheetBinding? = null
    
    private var ngWeight = 1.0f
    private var isDecoderDebugEnabled = false
    private var decoderMode = MyScriptService.DecoderMode.LLM
    private var llmTimeoutMs = 1000L
    private var llmModelIndex = 0
    private var scraperTargetHeight = 720
    
    private val camFpsQueue = java.util.ArrayDeque<Long>()
    private val mpFpsQueue = java.util.ArrayDeque<Long>()
    
    private val isProcessingFrame = java.util.concurrent.atomic.AtomicBoolean(false)
    private val isCopyingFrame = java.util.concurrent.atomic.AtomicBoolean(false)
    private var bufferWriting: Bitmap? = null
    private var bufferReady: Bitmap? = null
    private var bufferReading: Bitmap? = null
    private var isNewFrameReady = false
    private val bufferLock = Any()
    private var pixelCopyThread: android.os.HandlerThread? = null
    private var pixelCopyHandler: android.os.Handler? = null
    
    // Jitter Buffer / Pacer
    private var isJitterBufferEnabled = false
    private val pacerQueue = java.util.LinkedList<Bitmap>()
    private val pacerPool = java.util.LinkedList<Bitmap>()
    private val pacerHandler = Handler(Looper.getMainLooper())
    private var pacerRunnable: Runnable? = null

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the HandLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if (handLandmarkerHelper.isClose()) {
                handLandmarkerHelper.setupHandLandmarker()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::handLandmarkerHelper.isInitialized) {
            viewModel.setMaxHands(handLandmarkerHelper.maxNumHands)
            viewModel.setMinHandDetectionConfidence(handLandmarkerHelper.minHandDetectionConfidence)
            viewModel.setMinHandTrackingConfidence(handLandmarkerHelper.minHandTrackingConfidence)
            viewModel.setMinHandPresenceConfidence(handLandmarkerHelper.minHandPresenceConfidence)
            viewModel.setDelegate(handLandmarkerHelper.currentDelegate)

            // Close the HandLandmarkerHelper and release resources
            backgroundExecutor.execute { handLandmarkerHelper.clearHandLandmarker() }
        }
        
        // Stop Smart Glasses if running
        smartGlassesService?.disconnect()
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        bottomSheetBinding = null
        super.onDestroyView()
        
        myScriptService?.close()
        smartGlassesService?.disconnect()
        
        pixelCopyThread?.quitSafely()
        pixelCopyThread = null
        pixelCopyHandler = null
        
        if (tts != null) {
            tts?.stop()
            tts?.shutdown()
        }

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()
        
        // Initialize PixelCopy thread
        pixelCopyThread = android.os.HandlerThread("PixelCopyThread").apply { start() }
        pixelCopyHandler = android.os.Handler(pixelCopyThread!!.looper)

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }
        
        // Initialize TTS
        tts = TextToSpeech(context, this)

        // Create the HandLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            // Check if delegate was set, otherwise default to GPU (1)
            val currentDelegate = if (viewModel.currentDelegate == HandLandmarkerHelper.DELEGATE_CPU) {
                 HandLandmarkerHelper.DELEGATE_GPU
            } else {
                 viewModel.currentDelegate
            }
            viewModel.setDelegate(currentDelegate) // Update VM

            handLandmarkerHelper = HandLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minHandDetectionConfidence = viewModel.currentMinHandDetectionConfidence,
                minHandTrackingConfidence = viewModel.currentMinHandTrackingConfidence,
                minHandPresenceConfidence = viewModel.currentMinHandPresenceConfidence,
                maxNumHands = viewModel.currentMaxHands,
                currentDelegate = currentDelegate,
                handLandmarkerHelperListener = this
            )
        }
        
        // Initialize Smart Glasses Service
        smartGlassesService = SmartGlassesStreamService(object : SmartGlassesStreamService.StreamListener {
            override fun onFrameReceived(bitmap: Bitmap) {
                updateFpsCounter(camFpsQueue, fragmentCameraBinding.textCameraFps, "Cam FPS")
                // 1. Update UI
                activity?.runOnUiThread {
                    fragmentCameraBinding.smartGlassesView.setImageBitmap(bitmap)
                }
                
                // 2. Run Inference (Background)
                if (this@CameraFragment::backgroundExecutor.isInitialized && !backgroundExecutor.isShutdown) {
                    backgroundExecutor.execute {
                        if (!isBlinking) {
                            handLandmarkerHelper.detectLiveStreamBitmap(bitmap, isSmartGlassesFlipped, isSmartGlassesMirrored)
                        }
                    }
                }
            }
            
            override fun onH264NalUnitReceived(data: ByteArray) {
                h264Decoder?.decodeNalUnit(data)
            }

            override fun onConnectionStatusChanged(isConnected: Boolean, message: String?) {
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), message ?: "Connection Status: $isConnected", Toast.LENGTH_SHORT).show()
                }
            }
        })
        
        // Initialize MyScript
        myScriptService = MyScriptService(requireContext(), object : MyScriptService.RecognitionListener {
            override fun onTextRecognized(text: String, debugText: String) {
                 activity?.runOnUiThread {
                     fragmentCameraBinding.textRecognitionResult.text = text
                     if (isDecoderDebugEnabled && debugText.isNotEmpty()) {
                         fragmentCameraBinding.textNgramDebug.text = debugText
                         fragmentCameraBinding.textNgramDebug.visibility = View.VISIBLE
                     } else {
                         fragmentCameraBinding.textNgramDebug.visibility = View.GONE
                     }
                     
                     // Speak only if drawing mode is active and text is not empty
                     if (isDrawingMode && text.isNotBlank()) {
                         tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "UtteranceId")
                     }
                 }
            }
            
            override fun onJiixReceived(items: List<Item>) {
                activity?.runOnUiThread {
                    fragmentCameraBinding.inkPreview.setStrokes(items)
                }
            }
        })
        myScriptService?.setDisplayMetrics(resources.displayMetrics)
        myScriptService?.decoderMode = decoderMode
        myScriptService?.setDebugMode(isDecoderDebugEnabled)
        myScriptService?.llmTimeoutMs = llmTimeoutMs
        myScriptService?.setLlmModelIndex(llmModelIndex)
        myScriptService?.preloadLlm()
        
        fragmentCameraBinding.overlay.strokeListener = object : OverlayView.OnStrokeListener {
            override fun onStroke(points: List<MyScriptService.PointData>) {
                myScriptService?.addStroke(points)
            }
            override fun onClear() {
                // When user clears drawing, trigger recognition commit
                myScriptService?.commitAndClear()
            }
            override fun onDoublePinch() {
                // Undo last word from LLM context
                val didUndo = myScriptService?.undoLastWord() == true
                if (didUndo) {
                    tts?.speak("I mean", TextToSpeech.QUEUE_FLUSH, null, "DoublePinchUndo")
                }

                // Silent Clear - visually remove dots 0.1s later
                fragmentCameraBinding.overlay.postDelayed({
                    fragmentCameraBinding.overlay.clearDrawing()
                    fragmentCameraBinding.overlay.invalidate()
                }, 100)
            }
            override fun onTriplePinch() {
                // Clear the canvas visually 0.1s later (no context change)
                fragmentCameraBinding.overlay.postDelayed({
                    fragmentCameraBinding.overlay.clearDrawing()
                    fragmentCameraBinding.overlay.invalidate()
                }, 100)
            }
            override fun onDebugCoords(x: Float, y: Float) {
                activity?.runOnUiThread {
                    fragmentCameraBinding.textDebugCoords.text = "X: ${x.toInt()}, Y: ${y.toInt()}"
                }
            }
        }

        updateWideAngleButtonVisibility()

        fragmentCameraBinding.btnMirrorCamera.setOnClickListener {
            if (isSmartGlassesMode) {
                isSmartGlassesMirrored = !isSmartGlassesMirrored
                val rotation = if (isSmartGlassesMirrored) 180f else 0f
                val scale = if (isSmartGlassesMirrored) -1f else 1f
                
                // MJPEG and TextureView work with standard rotations
                fragmentCameraBinding.smartGlassesView.rotationY = rotation
                fragmentCameraBinding.smartGlassesH264View.rotationY = rotation
                
                // RTSP SurfaceView: Apply BOTH scale and rotation to the container and the view
                // This is the most aggressive way to force the Hardware Composer to flip
                fragmentCameraBinding.smartGlassesRtspContainer.scaleX = scale
                fragmentCameraBinding.smartGlassesRtspView.scaleX = scale
                fragmentCameraBinding.smartGlassesRtspView.rotationY = rotation
            }
        }

        fragmentCameraBinding.btnFlipCamera.setOnClickListener {
            if (isSmartGlassesMode) {
                isSmartGlassesFlipped = !isSmartGlassesFlipped
                val rotation = if (isSmartGlassesFlipped) 180f else 0f
                val scale = if (isSmartGlassesFlipped) -1f else 1f
                
                fragmentCameraBinding.smartGlassesView.rotationX = rotation
                fragmentCameraBinding.smartGlassesH264View.rotationX = rotation
                
                // RTSP SurfaceView: Apply BOTH scale and rotation to the container and the view
                fragmentCameraBinding.smartGlassesRtspContainer.scaleY = scale
                fragmentCameraBinding.smartGlassesRtspView.scaleY = scale
                fragmentCameraBinding.smartGlassesRtspView.rotationX = rotation
                return@setOnClickListener
            }
            
            cameraFacing = if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            updateWideAngleButtonVisibility()
            bindCameraUseCases()
        }

        fragmentCameraBinding.btnWideAngle.setOnClickListener {
            isWideAngle = !isWideAngle
            if (isWideAngle) {
                fragmentCameraBinding.btnWideAngle.text = "Normal"
            } else {
                fragmentCameraBinding.btnWideAngle.text = "Wide"
            }
            setZoom()
        }
        
        fragmentCameraBinding.btnDrawingMode.setOnClickListener {
            isDrawingMode = !isDrawingMode
            if (isDrawingMode) {
                fragmentCameraBinding.btnDrawingMode.text = "Draw: ON"
                fragmentCameraBinding.textRecognitionResult.visibility = View.VISIBLE
                fragmentCameraBinding.textDebugCoords.visibility = View.GONE
            } else {
                fragmentCameraBinding.btnDrawingMode.text = "Draw: OFF"
                fragmentCameraBinding.textRecognitionResult.visibility = View.GONE
                fragmentCameraBinding.textDebugCoords.visibility = View.VISIBLE
            }
            fragmentCameraBinding.overlay.isDrawingMode = isDrawingMode
        }

        fragmentCameraBinding.btnResetLlmContext.setOnClickListener {
            myScriptService?.resetLlmContext()
            Toast.makeText(requireContext(), "LLM context reset", Toast.LENGTH_SHORT).show()
        }
        
        fragmentCameraBinding.btnBlink.setOnClickListener {
            // Cut feed to MediaPipe by sending BLACK frames
            isBlinking = true
            fragmentCameraBinding.viewFinder.visibility = View.INVISIBLE
            fragmentCameraBinding.smartGlassesView.visibility = View.INVISIBLE // Also blink SG view if active
            
            fragmentCameraBinding.viewFinder.postDelayed({
                isBlinking = false
                if (isSmartGlassesMode) {
                    fragmentCameraBinding.smartGlassesView.visibility = View.VISIBLE
                } else {
                    fragmentCameraBinding.viewFinder.visibility = View.VISIBLE
                }
            }, 100)
        }
        
        fragmentCameraBinding.btnSettings.setOnClickListener {
            showSettingsDialog()
        }
        
        fragmentCameraBinding.btnSmartGlasses.setOnClickListener {
            toggleSmartGlassesMode()
        }
    }

    private fun startPacerIfNeeded() {
        if (pacerRunnable != null) return
        
        pacerRunnable = object : Runnable {
            override fun run() {
                if (!isJitterBufferEnabled || pacerQueue.isEmpty()) {
                    pacerRunnable = null
                    return
                }

                if (isProcessingFrame.compareAndSet(false, true)) {
                    val bitmapToProcess = synchronized(bufferLock) {
                        if (pacerQueue.isNotEmpty()) pacerQueue.removeFirst() else null
                    }

                    if (bitmapToProcess != null) {
                        backgroundExecutor.execute {
                            try {
                                if (!isBlinking) {
                                    handLandmarkerHelper.detectLiveStreamBitmap(bitmapToProcess, isSmartGlassesFlipped, isSmartGlassesMirrored)
                                }
                                // Return the bitmap to the pool when done
                                synchronized(bufferLock) {
                                    pacerPool.add(bitmapToProcess)
                                }
                                isProcessingFrame.set(false)
                            } catch (e: Exception) {
                                Log.e(TAG, "Error in pacer processing", e)
                                synchronized(bufferLock) {
                                    pacerPool.add(bitmapToProcess)
                                }
                                isProcessingFrame.set(false)
                            }
                        }
                    } else {
                        isProcessingFrame.set(false)
                    }
                }
                
                pacerHandler.postDelayed(this, 33) // Steady 30fps pacing
            }
        }
        pacerHandler.post(pacerRunnable!!)
    }
    
    private fun toggleSmartGlassesMode() {
        if (!isSmartGlassesMode) {
            // Enable Smart Glasses Mode
            val layout = android.widget.LinearLayout(requireContext()).apply { 
                orientation = android.widget.LinearLayout.VERTICAL
                setPadding(50, 40, 50, 10) 
            }
            
            val input = android.widget.EditText(requireContext())
            input.setText(lastSocketUrl)
            
            val radioGroup = android.widget.RadioGroup(requireContext()).apply {
                orientation = android.widget.RadioGroup.VERTICAL
                setPadding(0, 20, 0, 0)
            }
            
            val rbClassic = android.widget.RadioButton(requireContext()).apply {
                text = "Classic (MJPEG)"
                id = android.view.View.generateViewId()
            }
            val rbH264Classic = android.widget.RadioButton(requireContext()).apply {
                text = "H.264 Classic (NAL)"
                id = android.view.View.generateViewId()
            }
            val rbRtsp = android.widget.RadioButton(requireContext()).apply {
                text = "H.264 RTSP (UDP)"
                id = android.view.View.generateViewId()
            }
            
            radioGroup.addView(rbClassic)
            radioGroup.addView(rbH264Classic)
            radioGroup.addView(rbRtsp)
            
            // Default selection based on current/last state
            when(currentStreamMode) {
                MODE_CLASSIC -> rbClassic.isChecked = true
                MODE_H264_CLASSIC -> rbH264Classic.isChecked = true
                MODE_H264_RTSP -> rbRtsp.isChecked = true
            }
            
            radioGroup.setOnCheckedChangeListener { _, checkedId ->
                val ip = lastSocketUrl.substringAfter("//").substringBefore(":")
                when(checkedId) {
                    rbClassic.id -> {
                        input.setText("ws://$ip:81")
                    }
                    rbH264Classic.id -> {
                        input.setText("ws://$ip:81")
                    }
                    rbRtsp.id -> {
                        input.setText("rtsp://192.168.1.251:554/stream")
                    }
                }
            }
            
            layout.addView(input)
            layout.addView(radioGroup)
            
            android.app.AlertDialog.Builder(requireContext())
                .setTitle("Connect to Smart Glasses")
                .setView(layout)
                .setPositiveButton("Connect") { _, _ ->
                    val url = input.text.toString()
                    if (url.isNotBlank()) {
                        lastSocketUrl = url
                        val mode = when(radioGroup.checkedRadioButtonId) {
                            rbH264Classic.id -> MODE_H264_CLASSIC
                            rbRtsp.id -> MODE_H264_RTSP
                            else -> MODE_CLASSIC
                        }
                        enableSmartGlasses(url, mode)
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            // Disable Smart Glasses Mode
            disableSmartGlasses()
        }
    }
    
    private fun enableSmartGlasses(url: String, mode: Int) {
        isSmartGlassesMode = true
        currentStreamMode = mode
        
        // 1. Unbind local camera
        cameraProvider?.unbindAll()
        fragmentCameraBinding.viewFinder.visibility = View.INVISIBLE
        
        // 1.5 Prepare Decoder/Player
        if (mode == MODE_H264_CLASSIC) {
            fragmentCameraBinding.smartGlassesH264View.visibility = View.VISIBLE
            fragmentCameraBinding.smartGlassesView.visibility = View.GONE
            fragmentCameraBinding.smartGlassesRtspView.visibility = View.GONE
            
            fragmentCameraBinding.smartGlassesH264View.surfaceTextureListener = object : android.view.TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(surfaceTexture: android.graphics.SurfaceTexture, width: Int, height: Int) {
                    bufferWriting = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    bufferReady = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    bufferReading = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    val surface = android.view.Surface(surfaceTexture)
                    
                    h264Decoder = H264Decoder(surface).apply {
                        onFrameRenderedListener = {
                            handleDecodedFrame(surface)
                        }
                        start()
                    }
                    smartGlassesService?.connect(url, true)
                }

                override fun onSurfaceTextureSizeChanged(surface: android.graphics.SurfaceTexture, width: Int, height: Int) {}
                override fun onSurfaceTextureDestroyed(surface: android.graphics.SurfaceTexture): Boolean {
                    h264Decoder?.stop()
                    h264Decoder = null
                    return true
                }
                override fun onSurfaceTextureUpdated(surface: android.graphics.SurfaceTexture) {}
            }
        } else if (mode == MODE_H264_RTSP) {
            fragmentCameraBinding.smartGlassesRtspContainer.visibility = View.VISIBLE
            fragmentCameraBinding.smartGlassesH264View.visibility = View.GONE
            fragmentCameraBinding.smartGlassesView.visibility = View.GONE
            
            setupRtspPlayer(url)
        } else {
            fragmentCameraBinding.smartGlassesView.visibility = View.VISIBLE
            fragmentCameraBinding.smartGlassesH264View.visibility = View.GONE
            
            // Connect immediately for MJPEG
            smartGlassesService?.connect(url, false)
        }
        
        fragmentCameraBinding.btnMirrorCamera.visibility = View.VISIBLE
        
        // Update Button visual state (optional)
        fragmentCameraBinding.btnSmartGlasses.setColorFilter(androidx.core.content.ContextCompat.getColor(requireContext(), com.google.mediapipe.examples.handlandmarker.R.color.mp_color_primary))
    }

    private fun handleDecodedFrame(surface: android.view.Surface) {
        // Update cam FPS to reflect physical decode rate
        updateFpsCounter(camFpsQueue, fragmentCameraBinding.textCameraFps, "Cam FPS")
        
        // Decoupled Scraper: Continuously write to bufferWriting at 30fps
        if (this@CameraFragment::backgroundExecutor.isInitialized && !backgroundExecutor.isShutdown && bufferWriting != null && pixelCopyHandler != null) {
            if (isCopyingFrame.compareAndSet(false, true)) {
                android.view.PixelCopy.request(
                    surface,
                    bufferWriting!!,
                    { copyResult ->
                        isCopyingFrame.set(false)
                        if (copyResult == android.view.PixelCopy.SUCCESS) {
                            // Swap with the ready buffer so Tracker can grab the freshest frame
                            synchronized(bufferLock) {
                                val temp = bufferReady
                                bufferReady = bufferWriting
                                bufferWriting = temp
                                isNewFrameReady = true
                            }
                            // Poke the tracker
                            triggerInference()
                        }
                    },
                    pixelCopyHandler!!
                )
            }
        }
    }

    private fun triggerInference() {
        if (isJitterBufferEnabled) {
            // Smooth Mode: Add to queue using pointer swapping (Zero Allocation)
            synchronized(bufferLock) {
                if (isNewFrameReady) {
                    bufferReady?.let { source ->
                        // Get a spare bitmap from the pool
                        val pacedBitmap = if (pacerPool.isNotEmpty()) pacerPool.removeFirst() else Bitmap.createBitmap(source)
                        
                        // Copy the pixels efficiently (still faster than createBitmap)
                        val canvas = android.graphics.Canvas(pacedBitmap)
                        canvas.drawBitmap(source, 0f, 0f, null)
                        
                        pacerQueue.add(pacedBitmap)
                        // Keep queue small to prevent excessive latency (max 3 frames ~100ms)
                        while (pacerQueue.size > 3) {
                            pacerPool.add(pacerQueue.removeFirst())
                        }
                    }
                    isNewFrameReady = false
                }
            }
            startPacerIfNeeded()
            return
        }

        if (isProcessingFrame.compareAndSet(false, true)) {
            synchronized(bufferLock) {
                if (!isNewFrameReady) {
                    isProcessingFrame.set(false)
                    return
                }
                // Grab the freshest frame into the reading buffer
                val temp = bufferReading
                bufferReading = bufferReady
                bufferReady = temp
                isNewFrameReady = false
            }
            
            backgroundExecutor.execute {
                try {
                    if (!isBlinking && bufferReading != null) {
                        handLandmarkerHelper.detectLiveStreamBitmap(bufferReading!!, isSmartGlassesFlipped, isSmartGlassesMirrored)
                    } else {
                        isProcessingFrame.set(false)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in background processing", e)
                    isProcessingFrame.set(false)
                }
            }
        }
    }

    private fun setupRtspPlayer(url: String) {
        // We use the new RtspSurfaceView
        fragmentCameraBinding.smartGlassesRtspView.apply {
            // API 5.6.4 expects a Uri
            init(android.net.Uri.parse(url), "", "") 
            
            // The listener interface name and methods in 5.6.4
            setStatusListener(object : RtspStatusListener {
                override fun onRtspFirstFrameRendered() {
                    Log.d(TAG, "RTSP First Frame Rendered")
                    reinitScraperBitmaps()
                    startFrameScraping()
                }

                override fun onRtspFrameSizeChanged(width: Int, height: Int) {
                    Log.d(TAG, "RTSP Frame Size Changed: $width x $height")
                    adjustAspectRatio(width, height)
                    // Re-init bitmaps if aspect ratio changed significantly
                    reinitScraperBitmaps()
                }
            })
            
            // API 5.6.4 uses requestVideo/requestAudio
            start(requestVideo = true, requestAudio = false)
        }
    }

    private fun reinitScraperBitmaps() {
        val view = fragmentCameraBinding.smartGlassesRtspView
        // Use the layout dimensions to get the correct aspect ratio
        val layoutWidth = view.width
        val layoutHeight = view.height
        
        if (layoutWidth > 0 && layoutHeight > 0) {
            val ratio = layoutWidth.toFloat() / layoutHeight
            val targetH = scraperTargetHeight
            val targetW = (targetH * ratio).toInt()
            
            Log.d(TAG, "Initializing scraper bitmaps at $targetW x $targetH (Target ${targetH}p)")
            
            synchronized(bufferLock) {
                // Safely re-create the bitmaps
                bufferWriting = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
                bufferReady = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
                bufferReading = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
                isNewFrameReady = false
            }
        } else {
            // Fallback if view not laid out yet - retry in a bit
            view.post { reinitScraperBitmaps() }
        }
    }

    private fun startFrameScraping() {
        if (processingRunnable != null) return // Already running
        
        processingRunnable = object : Runnable {
            override fun run() {
                if (isSmartGlassesMode && currentStreamMode == MODE_H264_RTSP) {
                    val surface = fragmentCameraBinding.smartGlassesRtspView.holder.surface
                    // Unblock the scraper! Only prevent overlapping PixelCopies, let MediaPipe do its own thing.
                    if (surface.isValid && bufferWriting != null && !isCopyingFrame.get()) {
                        handleDecodedFrame(surface)
                    }
                }
                processingHandler.postDelayed(this, 33) // ~30fps polling
            }
        }
        processingHandler.post(processingRunnable!!)
    }

    private fun releaseRtspPlayer() {
        fragmentCameraBinding.smartGlassesRtspView.stop()
        processingRunnable?.let { processingHandler.removeCallbacks(it) }
        processingRunnable = null
    }

    

    private fun adjustAspectRatio(videoWidth: Int, videoHeight: Int) {
        val container = fragmentCameraBinding.smartGlassesH264View.parent as? View ?: return
        val containerWidth = container.width
        val containerHeight = container.height
        
        if (containerWidth == 0 || containerHeight == 0) return
        
        val aspectRatio = videoWidth.toFloat() / videoHeight
        
        // Calculate dimensions to fill the width of the phone
        val targetWidth = containerWidth
        val targetHeight = (containerWidth / aspectRatio).toInt()
        
        // Ensure we don't exceed container height (rare on phones for landscape video)
        val finalWidth: Int
        val finalHeight: Int
        if (targetHeight > containerHeight) {
            finalHeight = containerHeight
            finalWidth = (containerHeight * aspectRatio).toInt()
        } else {
            finalWidth = targetWidth
            finalHeight = targetHeight
        }
        
        // Update LayoutParams for ALL relevant views to ensure pixel-perfect alignment
        fragmentCameraBinding.cameraContainer.post {
            // Update H264 TextureView (Classic)
            val h264Params = fragmentCameraBinding.smartGlassesH264View.layoutParams
            h264Params.width = finalWidth
            h264Params.height = finalHeight
            fragmentCameraBinding.smartGlassesH264View.layoutParams = h264Params
            
            // Update RTSP SurfaceView Container
            val rtspContainerParams = fragmentCameraBinding.smartGlassesRtspContainer.layoutParams
            rtspContainerParams.width = finalWidth
            rtspContainerParams.height = finalHeight
            fragmentCameraBinding.smartGlassesRtspContainer.layoutParams = rtspContainerParams
            
            // Update OverlayView
            val overlayParams = fragmentCameraBinding.overlay.layoutParams
            overlayParams.width = finalWidth
            overlayParams.height = finalHeight
            fragmentCameraBinding.overlay.layoutParams = overlayParams
            
            // Remove any previously applied matrix transforms
            fragmentCameraBinding.smartGlassesH264View.setTransform(null)
        }
    }


    
    private fun disableSmartGlasses() {
        isSmartGlassesMode = false
        // 1. Disconnect
        smartGlassesService?.disconnect()
        
        h264Decoder?.stop()
        h264Decoder = null
        
        releaseRtspPlayer()
        
        // 2. Hide SG View
        fragmentCameraBinding.smartGlassesView.setImageBitmap(null)
        fragmentCameraBinding.smartGlassesView.visibility = View.GONE
        fragmentCameraBinding.smartGlassesH264View.visibility = View.GONE
        fragmentCameraBinding.smartGlassesRtspContainer.visibility = View.GONE
        fragmentCameraBinding.btnMirrorCamera.visibility = View.GONE
        
        // 3. Rebind local camera
        fragmentCameraBinding.viewFinder.visibility = View.VISIBLE
        bindCameraUseCases()
        
        // Update Button visual state (optional)
        fragmentCameraBinding.btnSmartGlasses.clearColorFilter()
    }
    
    private fun showSettingsDialog() {
        if (settingsDialog == null) {
            settingsDialog = BottomSheetDialog(requireContext())
            bottomSheetBinding = InfoBottomSheetBinding.inflate(layoutInflater)
            settingsDialog?.setContentView(bottomSheetBinding!!.root)
            initBottomSheetControls()
        }
        updateControlsUi()
        settingsDialog?.show()
    }
    
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "Language not supported")
            }
        } else {
            Log.e("TTS", "Initialization failed")
        }
    }

    private fun updateWideAngleButtonVisibility() {
        if (cameraFacing == CameraSelector.LENS_FACING_BACK) {
            fragmentCameraBinding.btnWideAngle.visibility = View.VISIBLE
        } else {
            fragmentCameraBinding.btnWideAngle.visibility = View.GONE
            // Reset wide angle state if we switch to front camera
            if (isWideAngle) {
                isWideAngle = false
                fragmentCameraBinding.btnWideAngle.text = "Wide"
            }
        }
    }

    private fun initBottomSheetControls() {
        if (bottomSheetBinding == null) return

        // init bottom sheet settings
        bottomSheetBinding!!.maxHandsValue.text =
            viewModel.currentMaxHands.toString()
        bottomSheetBinding!!.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandDetectionConfidence
            )
        bottomSheetBinding!!.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandTrackingConfidence
            )
        bottomSheetBinding!!.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandPresenceConfidence
            )

        // When clicked, lower hand detection score threshold floor
        bottomSheetBinding!!.detectionThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence >= 0.2) {
                handLandmarkerHelper.minHandDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand detection score threshold floor
        bottomSheetBinding!!.detectionThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence <= 0.8) {
                handLandmarkerHelper.minHandDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand tracking score threshold floor
        bottomSheetBinding!!.trackingThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence >= 0.2) {
                handLandmarkerHelper.minHandTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand tracking score threshold floor
        bottomSheetBinding!!.trackingThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence <= 0.8) {
                handLandmarkerHelper.minHandTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand presence score threshold floor
        bottomSheetBinding!!.presenceThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence >= 0.2) {
                handLandmarkerHelper.minHandPresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand presence score threshold floor
        bottomSheetBinding!!.presenceThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence <= 0.8) {
                handLandmarkerHelper.minHandPresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of hands that can be detected at a
        // time
        bottomSheetBinding!!.maxHandsMinus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands > 1) {
                handLandmarkerHelper.maxNumHands--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of hands that can be detected
        // at a time
        bottomSheetBinding!!.maxHandsPlus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands < 2) {
                handLandmarkerHelper.maxNumHands++
                updateControlsUi()
            }
        }
        
        // N-Gram Weight controls
        bottomSheetBinding!!.ngramWeightMinus.setOnClickListener {
            if (ngWeight > 0.1f) {
                ngWeight -= 0.1f
                myScriptService?.setNgWeight(ngWeight)
                updateControlsUi()
            }
        }
        
        bottomSheetBinding!!.ngramWeightPlus.setOnClickListener {
            if (ngWeight < 1.95f) { // Float precision
                ngWeight += 0.1f
                myScriptService?.setNgWeight(ngWeight)
                updateControlsUi()
            }
        }

        // LLM Timeout controls
        bottomSheetBinding!!.llmTimeoutMinus.setOnClickListener {
            if (llmTimeoutMs > 500L) {
                llmTimeoutMs -= 100L
                myScriptService?.llmTimeoutMs = llmTimeoutMs
                updateControlsUi()
            }
        }

        bottomSheetBinding!!.llmTimeoutPlus.setOnClickListener {
            if (llmTimeoutMs < 5000L) {
                llmTimeoutMs += 100L
                myScriptService?.llmTimeoutMs = llmTimeoutMs
                updateControlsUi()
            }
        }
        
        bottomSheetBinding!!.ngramDebugSwitch.setOnCheckedChangeListener { _, isChecked ->
            isDecoderDebugEnabled = isChecked
            myScriptService?.setDebugMode(isChecked)
            if (!isChecked) {
                fragmentCameraBinding.textNgramDebug.visibility = View.GONE
            }
        }

        bottomSheetBinding!!.spinnerRecognitionMode.setSelection(getDecoderModeIndex(), false)
        bottomSheetBinding!!.spinnerRecognitionMode.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    decoderMode = when (p2) {
                        0 -> MyScriptService.DecoderMode.LLM
                        1 -> MyScriptService.DecoderMode.NGRAM
                        else -> MyScriptService.DecoderMode.NONE
                    }
                    myScriptService?.decoderMode = decoderMode
                    myScriptService?.llmTimeoutMs = llmTimeoutMs
                    if (decoderMode == MyScriptService.DecoderMode.LLM) {
                        myScriptService?.preloadLlm()
                    }
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        bottomSheetBinding!!.spinnerLlmModel.setSelection(llmModelIndex, false)
        bottomSheetBinding!!.spinnerLlmModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    llmModelIndex = p2
                    myScriptService?.setLlmModelIndex(p2)
                    if (decoderMode == MyScriptService.DecoderMode.LLM) {
                        myScriptService?.preloadLlm()
                    }
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        bottomSheetBinding!!.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        bottomSheetBinding!!.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        handLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "HandLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
            
        // Scraper resolution spinner
        val resOptions = listOf(360, 420, 540, 720, 1080)
        val currentResIndex = resOptions.indexOf(scraperTargetHeight).coerceAtLeast(0)
        bottomSheetBinding!!.spinnerScraperRes.setSelection(currentResIndex, false)
        bottomSheetBinding!!.spinnerScraperRes.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newHeight = resOptions[position]
                if (newHeight != scraperTargetHeight) {
                    scraperTargetHeight = newHeight
                    if (isSmartGlassesMode && currentStreamMode == MODE_H264_RTSP) {
                        reinitScraperBitmaps()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        bottomSheetBinding!!.jitterBufferSwitch.setOnCheckedChangeListener { _, isChecked ->
            isJitterBufferEnabled = isChecked
            if (!isChecked) {
                synchronized(bufferLock) {
                    pacerQueue.clear()
                }
            }
        }
        
        bottomSheetBinding!!.tapGesturesSwitch.isChecked = fragmentCameraBinding.overlay.isTapGesturesEnabled
        bottomSheetBinding!!.tapGesturesSwitch.setOnCheckedChangeListener { _, isChecked ->
            fragmentCameraBinding.overlay.isTapGesturesEnabled = isChecked
        }
        
        // Coordinate Scale buttons
        bottomSheetBinding!!.textScaleValue.text = String.format(java.util.Locale.US, "%.2f", fragmentCameraBinding.overlay.coordinateScale)
        bottomSheetBinding!!.btnScalePlus.setOnClickListener {
            fragmentCameraBinding.overlay.coordinateScale += 0.25f
            bottomSheetBinding!!.textScaleValue.text = String.format(java.util.Locale.US, "%.2f", fragmentCameraBinding.overlay.coordinateScale)
        }
        bottomSheetBinding!!.btnScaleMinus.setOnClickListener {
            fragmentCameraBinding.overlay.coordinateScale = (fragmentCameraBinding.overlay.coordinateScale - 0.25f).coerceAtLeast(0.25f)
            bottomSheetBinding!!.textScaleValue.text = String.format(java.util.Locale.US, "%.2f", fragmentCameraBinding.overlay.coordinateScale)
        }
    }

    private fun getDecoderModeIndex(): Int {
        return when (decoderMode) {
            MyScriptService.DecoderMode.LLM -> 0
            MyScriptService.DecoderMode.NGRAM -> 1
            MyScriptService.DecoderMode.NONE -> 2
        }
    }

    // Update the values displayed in the bottom sheet. Reset Handlandmarker
    // helper.
    private fun updateControlsUi() {
        if (bottomSheetBinding == null) return

        bottomSheetBinding!!.maxHandsValue.text =
            handLandmarkerHelper.maxNumHands.toString()
        bottomSheetBinding!!.detectionThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandDetectionConfidence
            )
        bottomSheetBinding!!.trackingThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandTrackingConfidence
            )
        bottomSheetBinding!!.presenceThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandPresenceConfidence
            )
            
        bottomSheetBinding!!.ngramWeightValue.text = String.format(Locale.US, "%.1f", ngWeight)
        bottomSheetBinding!!.ngramDebugSwitch.isChecked = isDecoderDebugEnabled
        bottomSheetBinding!!.spinnerRecognitionMode.setSelection(getDecoderModeIndex(), false)
        bottomSheetBinding!!.spinnerLlmModel.setSelection(llmModelIndex, false)

        val isNgramMode = decoderMode == MyScriptService.DecoderMode.NGRAM
        bottomSheetBinding!!.ngramWeightRow.visibility = if (isNgramMode) View.VISIBLE else View.GONE

        val isLlmMode = decoderMode == MyScriptService.DecoderMode.LLM
        bottomSheetBinding!!.llmTimeoutRow.visibility = if (isLlmMode) View.VISIBLE else View.GONE
        bottomSheetBinding!!.llmTimeoutValue.text = String.format(Locale.US, "%.1f", llmTimeoutMs / 1000f)
        bottomSheetBinding!!.llmModelRow.visibility = if (isLlmMode) View.VISIBLE else View.GONE

        fragmentCameraBinding.btnResetLlmContext.visibility =
            if (isLlmMode) View.VISIBLE else View.GONE

        fragmentCameraBinding.textLlmStatus.visibility =
            if (isLlmMode) View.VISIBLE else View.GONE
            
        if(this::handLandmarkerHelper.isInitialized) {
            bottomSheetBinding!!.inferenceTimeVal.text = "0 ms" // Reset or update from helper if possible
        }

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        backgroundExecutor.execute {
            handLandmarkerHelper.clearHandLandmarker()
            handLandmarkerHelper.setupHandLandmarker()
        }
        fragmentCameraBinding.overlay.clear()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        updateFpsCounter(camFpsQueue, fragmentCameraBinding.textCameraFps, "Cam FPS")
                        if (!isBlinking) {
                            detectHand(image)
                        } else {
                            // Close image if we are blinking to avoid memory leak
                            image.close()
                        }
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
            cameraControl = camera?.cameraControl

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
            
            setZoom()
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }
    
    private fun setZoom() {
        if (cameraFacing == CameraSelector.LENS_FACING_BACK) {
            // 0.0f is wide (no zoom), 0.5f is "normal" (zoomed in)
            cameraControl?.setLinearZoom(if (isWideAngle) 0.0f else 0.5f)
        } else {
            cameraControl?.setLinearZoom(0.0f)
        }
    }

    private fun detectHand(imageProxy: ImageProxy) {
        // Updated to pass isBlinking
        handLandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT,
            isBlackout = isBlinking
        )
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after hand have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: HandLandmarkerHelper.ResultBundle
    ) {
        isProcessingFrame.set(false)
        triggerInference() // Pull next frame immediately if ready
        
        updateFpsCounter(mpFpsQueue, fragmentCameraBinding.textMediapipeFps, "MP FPS")
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                // Remove direct update of bottom sheet here, as it's now in a dialog
                // If the dialog is open, update it
                if (settingsDialog?.isShowing == true && bottomSheetBinding != null) {
                     bottomSheetBinding!!.inferenceTimeVal.text =
                        String.format("%d ms", resultBundle.inferenceTime)
                }

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )
                
                // Show confidence score
                if (resultBundle.results.isNotEmpty()) {
                    val result = resultBundle.results.first()
                    if (result.handedness().isNotEmpty()) {
                        val score = result.handedness().first().first().score()
                        fragmentCameraBinding.textConfidenceDebug.text = String.format("Conf: %.2f", score)
                    }
                } else {
                    fragmentCameraBinding.textConfidenceDebug.text = "Conf: --"
                }

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
                
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        isProcessingFrame.set(false)
        triggerInference() // Pull next frame immediately if ready
        
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == HandLandmarkerHelper.GPU_ERROR) {
                // If dialog is open, update spinner
                if (settingsDialog?.isShowing == true && bottomSheetBinding != null) {
                    bottomSheetBinding!!.spinnerDelegate.setSelection(
                        HandLandmarkerHelper.DELEGATE_CPU, false
                    )
                }
            }
        }
    }

    private fun updateFpsCounter(queue: java.util.ArrayDeque<Long>, textView: android.widget.TextView, prefix: String) {
        val now = System.currentTimeMillis()
        queue.addLast(now)
        while (queue.isNotEmpty() && now - queue.first() > 5000L) {
            queue.removeFirst()
        }
        val count = queue.size
        val duration = if (count > 1) (now - queue.first()) / 1000f else 0f
        val fps = if (duration > 0) count / duration else 0f
        
        activity?.runOnUiThread {
            textView.text = String.format(java.util.Locale.US, "%s: %.1f", prefix, fps)
        }
    }
}
