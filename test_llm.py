import os
import sys

# Attempt to import mediapipe
# Attempt to import mediapipe with fallback paths
try:
    import mediapipe as mp
    try:
        from mediapipe.tasks.python.genai import llm_inference
    except ImportError:
        # Some versions hide it directly under tasks or skip the 'python' subfolder
        try:
            from mediapipe.tasks.genai import llm_inference
        except ImportError:
            import mediapipe.tasks.python.genai.llm_inference as llm_inference
except ImportError as e:
    print(f"\n[!] MediaPipe GenAI not found. Error: {e}")
    print("Version 0.10.24+ is required on Windows for GenAI support.")
    print("Please run: pip install --upgrade mediapipe")
    sys.exit(1)

MODEL_PATH = "llm.bin"  # Ensure your gemma-1.1-2b-it-gpu-int4.bin is renamed to llm.bin here

def clean_text(text):
    """Simple cleanup to match candidate list"""
    return "".join(c for c in text if c.isalnum()).strip().lower()

class LlmTester:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"\n[!] Model file NOT FOUND: {model_path}")
            print("Please download Gemma 2B from Kaggle, rename it to llm.bin, and place it in this folder.")
            sys.exit(1)

        print(f"--- Initializing LLM from {model_path} ---")
        options = llm_inference.LlmInferenceOptions(
            model_path=model_path,
            max_tokens=128,  # We only need short answers
            top_k=40,
            temperature=0.2, # Low temperature for consistent selection
        )
        self.model = llm_inference.LlmInference.create_from_options(options)
        print("--- LLM Ready ---\n")

    def get_selection(self, context, candidates):
        # Mirroring the Kotlin logic:
        # Task: Pick the right word. Context: '{context}'. Options: [{candidates}]. Output only the exact word.
        candidates_str = ", ".join(candidates)
        prompt = f"Task: Pick the right word. Context: '{context}'. Options: [{candidates_str}]. Output only the exact word."
        
        print(f"Testing Context: \"{context}\"")
        print(f"Candidates: {candidates}")
        
        response = self.model.generate_response(prompt)
        clean_response = clean_text(response)
        
        print(f"LLM Raw Output: \"{response.strip()}\"")
        
        # Match against our original list
        matched = None
        for c in candidates:
            if clean_response == clean_text(c):
                matched = c
                break
        
        if matched:
            print(f"==> SELECTED: {matched} [OK]")
        else:
            print(f"==> ERROR: Hallucination! LLM output doesn't match candidates.")
        print("-" * 40)
        return matched

if __name__ == "__main__":
    tester = LlmTester(MODEL_PATH)

    # Test Case 1: Financial Context
    tester.get_selection(
        context="I went to the store today so I could deposit cash into the", 
        candidates=["bank", "bark", "bunk"]
    )

    # Test Case 2: Pet Context
    tester.get_selection(
        context="My golden retriever has a very loud and scary", 
        candidates=["bank", "bark", "bunk"]
    )

    # Test Case 3: Nautical Context
    tester.get_selection(
        context="I am tired and ready to go sleep in my tiny", 
        candidates=["bank", "bark", "bunk"]
    )
    
    # Test Case 4: Grammar check
    tester.get_selection(
        context="How many apples", 
        candidates=["are", "is", "as"]
    )
    
    # Test Case 5: Word choice
    tester.get_selection(
        context="I love you", 
        candidates=["too", "to", "two"]
    )
