package com.google.mediapipe.examples.handlandmarker

import android.app.Application
import android.util.Log
import com.myscript.certificate.MyCertificate
import com.myscript.iink.Engine
import java.io.File

class MyScriptApplication: Application() {

    var engine: Engine? = null

    override fun onCreate() {
        super.onCreate()

        try {
            engine = Engine.create(MyCertificate.getBytes()).apply {
                configuration.let { conf ->
                    val confDir = "zip://${packageCodePath}!/assets/conf"
                    conf.setStringArray("configuration-manager.search-path", arrayOf(confDir))
                    val tempDir = File(cacheDir, "tmp")
                    conf.setString("content-package.temp-folder", tempDir.absolutePath)
                    conf.setBoolean("offscreen-editor.history-manager.enable", true);
                }
            }
            Log.d(TAG, "MyScript Engine Initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MyScript Engine: ${e.message}")
        }
    }

    override fun onTerminate() {
        super.onTerminate()
        engine?.close()
        engine = null
    }

    companion object {
        private const val TAG = "MyScriptApplication"
    }
}
