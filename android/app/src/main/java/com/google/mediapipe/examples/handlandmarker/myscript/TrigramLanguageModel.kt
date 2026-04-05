package com.google.mediapipe.examples.handlandmarker.myscript

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.max
import kotlin.math.min
import kotlin.math.ln

data class DecodeResult(val text: String, val debugInfo: String)

class TrigramLanguageModel(private val context: Context) {

    private var db: SQLiteDatabase? = null
    var ngWeight = 1.0f
    var isDebugMode = false

    // History state tracking for sequential word inputs (one word at a time)
    private var historyW2: String = ""
    private var historyW1: String = ""
    
    // DB Extraction Flag
    @Volatile
    private var isDbReady = false
    
    // Relational caching to keep inference <= 15ms
    private val idCache = mutableMapOf<String, Int>()

    init {
        // We MUST run this huge database extraction on an IO thread
        // Otherwise, the Android UI will freeze on startup causing an ANR crash
        CoroutineScope(Dispatchers.IO).launch {
            openDatabase()
        }
    }

    private fun openDatabase() {
        try {
            val dbPath = context.getDatabasePath("ngrams.db")
            if (!dbPath.exists()) {
                dbPath.parentFile?.mkdirs()
                context.assets.open("ngrams.db").use { input ->
                    FileOutputStream(dbPath).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            db = SQLiteDatabase.openDatabase(dbPath.absolutePath, null, SQLiteDatabase.OPEN_READONLY)
            isDbReady = true
            Log.d(TAG, "Successfully loaded relational ngrams.db from SQLite")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ngrams.db: ${e.message}")
        }
    }

    /**
     * Clears the sequential word history context. Call this when starting a new completely distinct sentence.
     */
    fun resetHistory() {
        historyW2 = ""
        historyW1 = ""
    }

    fun decodeOptimalSentence(wordSequence: List<List<String>>): DecodeResult {
        if (wordSequence.isEmpty()) return DecodeResult("", "")
        
        val debugData = StringBuilder()
        
        // If DB is still copying in the background, don't crash, just gracefully 
        // fall back to MyScript's raw rank-0 suggestions!
        if (!isDbReady) {
            Log.w(TAG, "DB not ready! Falling back to raw MyScript output.")
            return DecodeResult(wordSequence.joinToString(" ") { it.firstOrNull() ?: "" }, "DB Loading...")
        }

        val trellis = ArrayList<MutableMap<Pair<String, String>, ViterbiNode>>()

        // Step 0: Evaluate the first word in the current burst using prior history
        val step0 = mutableMapOf<Pair<String, String>, ViterbiNode>()
        val firstWordCandidates = wordSequence[0].filter { it.isNotBlank() }
        
        if (isDebugMode) debugData.append("--- Word 1 ---\n")
        
        for ((rank, candidate) in firstWordCandidates.withIndex()) {
            val jiixScore = getJiixScore(rank)
            val currentLower = candidate.lowercase()
            
            // Calculate starting probability by blending history
            val lmProbRaw = if (historyW1.isNotBlank() && historyW2.isNotBlank()) {
                getTrigramLogProb(historyW2, historyW1, currentLower)
            } else if (historyW1.isNotBlank()) {
                getBigramLogProb(historyW1, currentLower)
            } else {
                getUnigramLogProb(currentLower)
            }
            
            val ngScore = normalizeScore(lmProbRaw)
            val score = jiixScore + (ngWeight * ngScore)
            
            if (isDebugMode) {
                debugData.append(String.format("%s (R%d): JIIX: %.2f | NG: %.2f | Tot: %.2f\n", candidate, rank, jiixScore, ngScore, score))
            }
            
            val stateKey = Pair(historyW1, currentLower)
            val existing = step0[stateKey]
            if (existing == null || score > existing.score) {
                step0[stateKey] = ViterbiNode(candidate, score, null)
            }
        }
        trellis.add(step0)

        // Step 1: Second word in burst uses historyW1 + Word 1
        if (wordSequence.size >= 2) {
            val step1 = mutableMapOf<Pair<String, String>, ViterbiNode>()
            val secondWordCandidates = wordSequence[1].filter { it.isNotBlank() }
            
            if (isDebugMode) debugData.append("\n--- Word 2 ---\n")
            
            for ((rank, currentCand) in secondWordCandidates.withIndex()) {
                val jiixScore = getJiixScore(rank)
                val currentLower = currentCand.lowercase()

                for ((prevCandKey, prevNode) in step0) {
                    val candidate0Lower = prevNode.word.lowercase()
                    
                    val transitionLogProb = if (historyW1.isNotBlank()) {
                        getTrigramLogProb(historyW1, candidate0Lower, currentLower)
                    } else {
                        getBigramLogProb(candidate0Lower, currentLower)
                    }
                    
                    val ngScore = normalizeScore(transitionLogProb)
                    val totalScore = prevNode.score + jiixScore + (ngWeight * ngScore)
                    val stateKey = Pair(candidate0Lower, currentLower)
                    
                    val existing = step1[stateKey]
                    if (existing == null || totalScore > existing.score) {
                        step1[stateKey] = ViterbiNode(currentCand, totalScore, prevNode)
                        if (isDebugMode && (existing == null || totalScore > existing.score)) {
                            // Only append debug for the single best path into this candidate to prevent spam
                            // For simplicity, just appending everything might be noisy, but useful
                        }
                    }
                }
                
                if (isDebugMode) {
                    val bestEntry = step1.entries.filter { it.value.word == currentCand }.maxByOrNull { it.value.score }
                    val finalScoreForCand = bestEntry?.value?.score ?: 0f
                    // Recalculate component for debug (rough)
                    val rawPrevScore = bestEntry?.value?.backpointer?.score ?: 0f
                    val thisStepAdd = finalScoreForCand - rawPrevScore
                    debugData.append(String.format("%s (R%d): JIIX: %.2f | Additive: %.2f | Tot: %.2f\n", currentCand, rank, jiixScore, thisStepAdd, finalScoreForCand))
                }
            }
            trellis.add(step1)
        }

        // Step 2+: Trigram transitions for words 3 to N within the same burst
        for (i in 2 until wordSequence.size) {
            val step = mutableMapOf<Pair<String, String>, ViterbiNode>()
            val candidates = wordSequence[i].filter { it.isNotBlank() }
            
            if (isDebugMode) debugData.append("\n--- Word ${i+1} ---\n")

            for ((rank, currentCand) in candidates.withIndex()) {
                val jiixScore = getJiixScore(rank)
                val currentLower = currentCand.lowercase()

                for ((prevStateKey, prevNode) in trellis[i - 1]) {
                    val wMinus2 = prevStateKey.first
                    val wMinus1 = prevStateKey.second
                    
                    val transitionLogProb = getTrigramLogProb(wMinus2, wMinus1, currentLower)
                    val ngScore = normalizeScore(transitionLogProb)
                    val totalScore = prevNode.score + jiixScore + (ngWeight * ngScore)
                    
                    val newStateKey = Pair(wMinus1, currentLower)
                    
                    val existing = step[newStateKey]
                    if (existing == null || totalScore > existing.score) {
                        step[newStateKey] = ViterbiNode(currentCand, totalScore, prevNode)
                    }
                }
                
                if (isDebugMode) {
                    val bestEntry = step.entries.filter { it.value.word == currentCand }.maxByOrNull { it.value.score }
                    val finalScoreForCand = bestEntry?.value?.score ?: 0f
                    val rawPrevScore = bestEntry?.value?.backpointer?.score ?: 0f
                    val thisStepAdd = finalScoreForCand - rawPrevScore
                    debugData.append(String.format("%s (R%d): JIIX: %.2f | Additive: %.2f | Tot: %.2f\n", currentCand, rank, jiixScore, thisStepAdd, finalScoreForCand))
                }
            }
            trellis.add(step)
        }

        val lastStep = trellis.last()
        var bestNode = lastStep.maxByOrNull { it.value.score }?.value
        
        val reversedPath = mutableListOf<String>()
        while (bestNode != null) {
            reversedPath.add(bestNode.word)
            bestNode = bestNode.backpointer
        }
        
        val finalSequence = reversedPath.reversed()
        
        // Update history state for the NEXT input based on what we just selected
        for (word in finalSequence) {
            val lowerWord = word.lowercase()
            historyW2 = historyW1
            historyW1 = lowerWord
        }

        return DecodeResult(finalSequence.joinToString(" "), debugData.toString())
    }

    private fun getJiixScore(rank: Int): Float {
        return max(0.2f, 1.0f - (rank * 0.2f))
    }

    private fun getWordId(word: String): Int? {
        if (db == null) return null
        
        // Use fast memory cache to avoid thrashing SQLite on repeated lookups during Viterbi
        idCache[word]?.let { return it }
        
        db?.rawQuery("SELECT id FROM dictionary WHERE word = ?", arrayOf(word))?.use { cursor ->
            if (cursor.moveToFirst()) {
                val id = cursor.getInt(0)
                idCache[word] = id
                return id
            }
        }
        return null
    }

    private fun getUnigramLogProb(w1: String): Float {
        if (db == null) return UNKNOWN_WORD_LOG_PROB
        val id1 = getWordId(w1) ?: return UNKNOWN_WORD_LOG_PROB
        
        db?.rawQuery("SELECT count FROM unigrams WHERE w1 = ?", arrayOf(id1.toString()))?.use { cursor ->
            if (cursor.moveToFirst()) {
                return ln(cursor.getInt(0).toFloat()) - TOTAL_CORPUS_LOG_PROB
            }
        }
        return UNKNOWN_WORD_LOG_PROB
    }

    private fun getBigramLogProb(w1: String, w2: String): Float {
        if (db == null) return BACKOFF_PENALTY + getUnigramLogProb(w2)
        val id1 = getWordId(w1)
        val id2 = getWordId(w2)
        
        if (id1 == null || id2 == null) return BACKOFF_PENALTY + getUnigramLogProb(w2)

        db?.rawQuery("SELECT count FROM bigrams WHERE w1 = ? AND w2 = ?", arrayOf(id1.toString(), id2.toString()))?.use { cursor ->
            if (cursor.moveToFirst()) {
                return ln(cursor.getInt(0).toFloat()) - TOTAL_CORPUS_LOG_PROB
            }
        }
        return BACKOFF_PENALTY + getUnigramLogProb(w2)
    }

    private fun getTrigramLogProb(w1: String, w2: String, w3: String): Float {
        if (db == null) return BACKOFF_PENALTY + getBigramLogProb(w2, w3)
        val id1 = getWordId(w1)
        val id2 = getWordId(w2)
        val id3 = getWordId(w3)
        
        if (id1 == null || id2 == null || id3 == null) return BACKOFF_PENALTY + getBigramLogProb(w2, w3)

        db?.rawQuery("SELECT count FROM trigrams WHERE w1 = ? AND w2 = ? AND w3 = ?", arrayOf(id1.toString(), id2.toString(), id3.toString()))?.use { cursor ->
            if (cursor.moveToFirst()) {
                return ln(cursor.getInt(0).toFloat()) - TOTAL_CORPUS_LOG_PROB
            }
        }
        return BACKOFF_PENALTY + getBigramLogProb(w2, w3)
    }

    fun close() {
        db?.close()
    }

    private data class ViterbiNode(
        val word: String,
        val score: Float,
        val backpointer: ViterbiNode?
    )

    companion object {
        private const val TAG = "TrigramLM"
        private const val UNKNOWN_WORD_LOG_PROB = -14.0f
        private const val BACKOFF_PENALTY = -2.0f
        private const val TOTAL_CORPUS_LOG_PROB = 17.5f // ~ln(40,000,000) for dynamic probability scaling
        
        fun normalizeScore(logProb: Float): Float {
            val minLog = -14.0f
            val maxLog = -3.0f
            val normalized = (logProb - minLog) / (maxLog - minLog)
            return max(0.0f, min(1.0f, normalized))
        }
    }
}
