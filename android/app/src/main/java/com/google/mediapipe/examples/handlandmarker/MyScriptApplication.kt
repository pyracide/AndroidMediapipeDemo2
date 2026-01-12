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
            val confDir = File(filesDir, "conf")
            val resDir = File(filesDir, "resources")

            // 1. Clean Slate
            if (confDir.exists()) confDir.deleteRecursively()
            if (resDir.exists()) resDir.deleteRecursively()
            
            // 2. Extract Resources First (so we have the target path)
            deployAssets("resources", resDir)
            
            // 3. Extract Conf
            deployAssets("conf", confDir)
            
            // 4. VERIFICATION DIAGNOSTICS
            verifyFile(File(resDir, "analyzer/ank-raw-content.res"))
            verifyFile(File(resDir, "document_layout/dl-raw-content.res"))
            verifyFile(File(confDir, "raw-content.conf"))
            
            // CHECK CUSTOM DICTIONARY
            verifyFile(File(resDir, "custom/10000_noletters.res"))

            // 5. Initialize Engine
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
    
    private fun verifyFile(file: File) {
        if (file.exists()) {
            Log.d(TAG, "VERIFIED: ${file.name} exists (Size: ${file.length()} bytes)")
        } else {
            Log.e(TAG, "CRITICAL MISSING: ${file.absolutePath} NOT FOUND")
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
