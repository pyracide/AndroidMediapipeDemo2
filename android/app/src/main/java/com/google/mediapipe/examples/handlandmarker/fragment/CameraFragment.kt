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

class CameraFragment : Fragment(), HandLandmarkerHelper.LandmarkerListener, TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "Hand Landmarker"
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
    
    private var myScriptService: MyScriptService? = null
    private var tts: TextToSpeech? = null
    private var settingsDialog: BottomSheetDialog? = null
    private var bottomSheetBinding: InfoBottomSheetBinding? = null
    
    private var ngWeight = 1.0f
    private var isNgDebugEnabled = false
    private var isNgramEnabled = true
    private var wasHandPresent = false
    
    private val camFpsQueue = java.util.ArrayDeque<Long>()
    private val mpFpsQueue = java.util.ArrayDeque<Long>()

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
                     if (isNgDebugEnabled && debugText.isNotEmpty()) {
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
        
        fragmentCameraBinding.overlay.strokeListener = object : OverlayView.OnStrokeListener {
            override fun onStroke(points: List<MyScriptService.PointData>) {
                myScriptService?.addStroke(points)
            }
            override fun onClear() {
                // When user clears drawing, trigger recognition commit
                myScriptService?.commitAndClear()
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
                fragmentCameraBinding.smartGlassesView.scaleX = if (isSmartGlassesMirrored) -1f else 1f
            }
        }

        fragmentCameraBinding.btnFlipCamera.setOnClickListener {
            if (isSmartGlassesMode) {
                isSmartGlassesFlipped = !isSmartGlassesFlipped
                fragmentCameraBinding.smartGlassesView.scaleY = if (isSmartGlassesFlipped) -1f else 1f
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
    
    private fun toggleSmartGlassesMode() {
        if (!isSmartGlassesMode) {
            // Enable Smart Glasses Mode
            val input = EditText(requireContext())
            input.setText(lastSocketUrl)
            
            AlertDialog.Builder(requireContext())
                .setTitle("Connect to Websocket")
                .setView(input)
                .setPositiveButton("Connect") { _, _ ->
                    val url = input.text.toString()
                    if (url.isNotBlank()) {
                        lastSocketUrl = url
                        enableSmartGlasses(url)
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            // Disable Smart Glasses Mode
            disableSmartGlasses()
        }
    }
    
    private fun enableSmartGlasses(url: String) {
        isSmartGlassesMode = true
        // 1. Unbind local camera
        cameraProvider?.unbindAll()
        fragmentCameraBinding.viewFinder.visibility = View.INVISIBLE
        
        // 2. Show SG View
        fragmentCameraBinding.smartGlassesView.visibility = View.VISIBLE
        fragmentCameraBinding.btnMirrorCamera.visibility = View.VISIBLE
        
        // 3. Connect
        smartGlassesService?.connect(url)
        
        // Update Button visual state (optional)
        fragmentCameraBinding.btnSmartGlasses.setColorFilter(ContextCompat.getColor(requireContext(), R.color.mp_color_primary))
    }
    
    private fun disableSmartGlasses() {
        isSmartGlassesMode = false
        // 1. Disconnect
        smartGlassesService?.disconnect()
        
        // 2. Hide SG View
        fragmentCameraBinding.smartGlassesView.setImageBitmap(null)
        fragmentCameraBinding.smartGlassesView.visibility = View.GONE
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
        
        bottomSheetBinding!!.ngramDebugSwitch.setOnCheckedChangeListener { _, isChecked ->
            isNgDebugEnabled = isChecked
            myScriptService?.setNgDebugMode(isChecked)
            if (!isChecked) {
                fragmentCameraBinding.textNgramDebug.visibility = View.GONE
            }
        }
        
        bottomSheetBinding!!.ngramEnableSwitch.setOnCheckedChangeListener { _, isChecked ->
            isNgramEnabled = isChecked
            myScriptService?.isNgramEnabled = isChecked
            if (!isChecked) {
                fragmentCameraBinding.textNgramDebug.visibility = View.GONE
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
        bottomSheetBinding!!.ngramDebugSwitch.isChecked = isNgDebugEnabled
        bottomSheetBinding!!.ngramEnableSwitch.isChecked = isNgramEnabled
            
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
                
                // Failsafe: if the hand is totally lost by ML, reset language history context
                val resultForTracking = resultBundle.results.firstOrNull()
                val isHandPresent = resultForTracking?.landmarks()?.isNotEmpty() == true
                
                if (!isHandPresent && wasHandPresent) {
                    myScriptService?.resetLanguageModelHistory()
                }
                wasHandPresent = isHandPresent
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
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
