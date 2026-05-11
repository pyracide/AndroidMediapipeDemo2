#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include "llama.cpp/include/llama.h"
#include "llama.cpp/common/common.h"

#define TAG "LLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model *model = nullptr;
static llama_context *ctx = nullptr;
static llama_batch batch;
static std::vector<llama_token> last_prompt_tokens;
static std::atomic<bool> cancel_flag(false);
static std::mutex llama_mutex;
static std::vector<int> last_token_counts;

static float get_log_prob(const float* logits, int vocab_size, llama_token token) {
    float max_logit = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return (logits[token] - max_logit) - logf(sum_exp);
}

static std::vector<llama_token> tokenize_prompt(const std::string &text) {
    std::vector<llama_token> tokens;
    tokens.resize(text.length() + 2);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
    }
    if (n_tokens > 0) {
        tokens.resize(n_tokens);
    } else {
        tokens.clear();
    }
    return tokens;
}

static std::vector<llama_token> tokenize_candidate(const std::string &text) {
    std::vector<llama_token> tokens;
    tokens.resize(text.length() + 2);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, false);
    }
    if (n_tokens > 0) {
        tokens.resize(n_tokens);
    } else {
        tokens.clear();
    }
    return tokens;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_loadModel(JNIEnv *env, jobject /* this */, jstring modelPath, jint numThreads) {
    std::lock_guard<std::mutex> lock(llama_mutex);
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model from %s", path);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;
    model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (model == nullptr) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.n_threads = numThreads;
    ctx_params.n_threads_batch = numThreads;

    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOGE("Failed to create context");
        return JNI_FALSE;
    }

    batch = llama_batch_init(512, 0, 1);
    cancel_flag.store(false);
    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_rankCandidatesNative(JNIEnv *env, jobject /* this */, jstring promptStr, jobjectArray candidatesArray) {
    std::lock_guard<std::mutex> lock(llama_mutex);
    const jsize num_candidates = env->GetArrayLength(candidatesArray);
    jfloatArray result = env->NewFloatArray(num_candidates);
    std::vector<float> scores(num_candidates, -1e9f);
    last_token_counts.assign(num_candidates, 0);

    if (model == nullptr || ctx == nullptr) {
        env->SetFloatArrayRegion(result, 0, num_candidates, scores.data());
        return result;
    }

    cancel_flag.store(false);

    const char *prompt_c = env->GetStringUTFChars(promptStr, nullptr);
    std::string prompt(prompt_c ? prompt_c : "");
    env->ReleaseStringUTFChars(promptStr, prompt_c);

    std::vector<llama_token> prompt_tokens = tokenize_prompt(prompt);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);

    size_t common_prefix_len = 0;
    while (common_prefix_len < std::min(prompt_tokens.size(), last_prompt_tokens.size()) &&
           prompt_tokens[common_prefix_len] == last_prompt_tokens[common_prefix_len]) {
        common_prefix_len++;
    }

    if (common_prefix_len == prompt_tokens.size() && common_prefix_len > 0) {
        common_prefix_len--;
    }

    if (common_prefix_len < last_prompt_tokens.size()) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, common_prefix_len, -1);
    }

    batch.n_tokens = 0;
    for (size_t i = common_prefix_len; i < prompt_tokens.size(); i++) {
        if (cancel_flag.load()) {
            env->SetFloatArrayRegion(result, 0, num_candidates, scores.data());
            return result;
        }
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }

    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
        if (llama_decode(ctx, batch) != 0) {
            LOGE("Failed to evaluate prompt");
            env->SetFloatArrayRegion(result, 0, num_candidates, scores.data());
            return result;
        }
    } else {
        env->SetFloatArrayRegion(result, 0, num_candidates, scores.data());
        return result;
    }

    batch.n_tokens = 0;
    last_prompt_tokens = prompt_tokens;

    float * logits = llama_get_logits_ith(ctx, -1);
    std::vector<float> prompt_logits(vocab_size);
    std::memcpy(prompt_logits.data(), logits, sizeof(float) * vocab_size);
    const bool use_sequence_scoring = true;

    for (jsize i = 0; i < num_candidates; i++) {
        if (cancel_flag.load()) {
            break;
        }
        jstring cand_jstr = (jstring) env->GetObjectArrayElement(candidatesArray, i);
        const char *cand_c = env->GetStringUTFChars(cand_jstr, nullptr);
        std::string cand(cand_c ? cand_c : "");
        env->ReleaseStringUTFChars(cand_jstr, cand_c);

        if (cand.empty()) {
            scores[i] = -1e9f;
            continue;
        }

        std::vector<std::string> variants = {cand, " " + cand};
        float best_score = -1e9f;
        int best_token_count = 0;

        for (const std::string& variant : variants) {
            std::vector<llama_token> c_toks = tokenize_candidate(variant);
            if (c_toks.empty()) {
                continue;
            }

            float score = 0.0f;
            if (!use_sequence_scoring) {
                score = prompt_logits[c_toks[0]];
            } else {
                score = get_log_prob(prompt_logits.data(), vocab_size, c_toks[0]);
                if (c_toks.size() > 1) {
                    const int seq_id = 1;
                    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, 0, -1);
                    llama_memory_seq_cp(llama_get_memory(ctx), 0, seq_id, 0, -1);

                    for (size_t t = 0; t < c_toks.size() - 1; t++) {
                        if (cancel_flag.load()) {
                            break;
                        }
                        batch.n_tokens = 0;
                        batch.token[0] = c_toks[t];
                        batch.pos[0] = prompt_tokens.size() + t;
                        batch.seq_id[0][0] = seq_id;
                        batch.n_seq_id[0] = 1;
                        batch.logits[0] = true;
                        batch.n_tokens = 1;

                        if (llama_decode(ctx, batch) != 0) {
                            LOGE("Failed to decode token %zu of candidate '%s'", t, variant.c_str());
                            break;
                        }
                        float* next_logits = llama_get_logits_ith(ctx, -1);
                        score += get_log_prob(next_logits, vocab_size, c_toks[t + 1]);
                    }

                    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, 0, -1);
                }
            }

            if (score > best_score) {
                best_score = score;
                best_token_count = static_cast<int>(c_toks.size());
            }
        }

        scores[i] = best_score;
        last_token_counts[i] = best_token_count;
    }

    env->SetFloatArrayRegion(result, 0, num_candidates, scores.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_resetNativeContext(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(llama_mutex);
    cancel_flag.store(true);
    last_prompt_tokens.clear();
    if (ctx) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
    }
    cancel_flag.store(false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_cancelNativeInference(JNIEnv *env, jobject /* this */) {
    cancel_flag.store(true);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_getLastTokenCountsNative(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(llama_mutex);
    jintArray result = env->NewIntArray(static_cast<jsize>(last_token_counts.size()));
    if (!last_token_counts.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(last_token_counts.size()), last_token_counts.data());
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_handlandmarker_myscript_LlmEngine_freeModel(JNIEnv *env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(llama_mutex);
    cancel_flag.store(true);
    last_prompt_tokens.clear();
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    llama_batch_free(batch);
    llama_backend_free();
    cancel_flag.store(false);
}
