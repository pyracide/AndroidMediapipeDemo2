package com.google.mediapipe.examples.handlandmarker

import android.app.Application
import android.util.Log
import com.myscript.certificate.MyCertificate
import com.myscript.iink.Engine
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MyScriptApplication: Application() {

    var engine: Engine? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "=== MyScriptApplication onCreate STARTED ===")

        try {
            // 1. Force Clean & Re-Extract Configuration
            val confDir = File(filesDir, "conf")
            if (confDir.exists()) confDir.deleteRecursively()
            deployAssets("conf", confDir)
            
            // 2. Force Clean & Re-Extract Resources (Language Packs)
            val resDir = File(filesDir, "resources")
            if (resDir.exists()) resDir.deleteRecursively()
            deployAssets("resources", resDir)
            
            // 3. DIAGNOSTIC: Verify critical files exist
            val analyzerFile = File(resDir, "analyzer/ank-raw-content.res")
            if (analyzerFile.exists()) {
                Log.d(TAG, "VERIFIED: analyzer/ank-raw-content.res exists")
            } else {
                Log.e(TAG, "CRITICAL MISSING: analyzer/ank-raw-content.res")
            }
            
            // NEW DIAGNOSTIC: Verify document_layout resource causing the IO_FAILURE
            val docLayoutFile = File(resDir, "document_layout/dl-raw-content.res")
            if (docLayoutFile.exists()) {
                Log.d(TAG, "VERIFIED: document_layout/dl-raw-content.res exists")
            } else {
                Log.e(TAG, "CRITICAL MISSING: document_layout/dl-raw-content.res NOT FOUND. This causes IO_FAILURE.")
                // List what IS in resources to help debug
                Log.d(TAG, "Contents of resources: ${resDir.list()?.joinToString()}")
                val layoutDir = File(resDir, "document_layout")
                if (layoutDir.exists()) {
                    Log.d(TAG, "Contents of document_layout: ${layoutDir.list()?.joinToString()}")
                } else {
                    Log.d(TAG, "document_layout directory does not exist")
                }
            }

            Log.d(TAG, "Attempting to create Engine instance...")
            engine = Engine.create(MyCertificate.getBytes())
            
            if (engine == null) {
                Log.e(TAG, "Engine.create() returned null (unexpected)")
                return
            }

            engine?.apply {
                configuration.let { conf ->
                    conf.setStringArray("configuration-manager.search-path", arrayOf(confDir.absolutePath))
                    
                    val tempDir = File(cacheDir, "tmp")
                    tempDir.mkdirs()
                    conf.setString("content-package.temp-folder", tempDir.absolutePath)
                    conf.setBoolean("offscreen-editor.history-manager.enable", true)
                }
            }
            Log.d(TAG, "=== MyScript Engine Initialized Successfully ===")
        } catch (t: Throwable) {
            Log.e(TAG, "FATAL FAILURE in Application.onCreate", t)
        }
    }
    
    private fun deployAssets(assetName: String, destination: File) {
        destination.mkdirs()
        
        val list = assets.list(assetName)
        if (list.isNullOrEmpty()) return 

        list.forEach { filename ->
            val subAsset = "$assetName/$filename"
            val destFile = File(destination, filename)
            
            val subList = assets.list(subAsset)
            
            if (subList != null && subList.isNotEmpty()) {
                deployAssets(subAsset, destFile)
            } else {
                try {
                    assets.open(subAsset).use { input ->
                        FileOutputStream(destFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    // Log.d(TAG, "Copied: $filename")
                } catch (e: IOException) {
                    Log.e(TAG, "Failed to copy asset: $subAsset", e)
                }
            }
        }
    }

    override fun onTerminate() {
        super.onTerminate()
        engine?.close()
        engine = null
    }

    companion object {
        private const val TAG = "Editor Logging"
    }
}
