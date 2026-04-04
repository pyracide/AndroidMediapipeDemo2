import sqlite3
import os
import math
import sys

# Constants matching the Kotlin implementation
UNKNOWN_WORD_LOG_PROB = -10.0
BACKOFF_PENALTY = -2.0
TOTAL_CORPUS_LOG_PROB = 17.5  # Roughly ln(40,000,000)

class NGramTester:
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            print(f"ERROR: Database not found at {db_path}.")
            print("Please wait for build_ngrams_sqlite.py to finish processing!")
            sys.exit(1)
        
        # Open in readonly mode to prevent locking issues if it's open elsewhere
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cursor = self.conn.cursor()
        self.id_cache = {}

    def get_word_id(self, word):
        word = word.lower().strip()
        if word in self.id_cache:
            return self.id_cache[word]
            
        self.cursor.execute("SELECT id FROM dictionary WHERE word = ?", (word,))
        row = self.cursor.fetchone()
        if row:
            self.id_cache[word] = row[0]
            return row[0]
        return None

    def get_unigram_log_prob(self, w1):
        id1 = self.get_word_id(w1)
        if id1 is None: return (UNKNOWN_WORD_LOG_PROB, "Unknown")
        
        self.cursor.execute("SELECT count FROM unigrams WHERE w1 = ?", (id1,))
        row = self.cursor.fetchone()
        if row:
            return (math.log(row[0]) - TOTAL_CORPUS_LOG_PROB, "Unigram")
        return (UNKNOWN_WORD_LOG_PROB, "Unknown")

    def get_bigram_log_prob(self, w1, w2):
        id1 = self.get_word_id(w1)
        id2 = self.get_word_id(w2)
        
        if id1 is None or id2 is None:
            score, eval_type = self.get_unigram_log_prob(w2)
            return (BACKOFF_PENALTY + score, eval_type)
            
        self.cursor.execute("SELECT count FROM bigrams WHERE w1 = ? AND w2 = ?", (id1, id2))
        row = self.cursor.fetchone()
        if row:
            return (math.log(row[0]) - TOTAL_CORPUS_LOG_PROB, "Bigram")
            
        score, eval_type = self.get_unigram_log_prob(w2)
        return (BACKOFF_PENALTY + score, eval_type)

    def get_trigram_log_prob(self, w1, w2, w3):
        id1 = self.get_word_id(w1)
        id2 = self.get_word_id(w2)
        id3 = self.get_word_id(w3)
        
        if id1 is None or id2 is None or id3 is None:
            score, eval_type = self.get_bigram_log_prob(w2, w3)
            return (BACKOFF_PENALTY + score, eval_type)
            
        self.cursor.execute("SELECT count FROM trigrams WHERE w1 = ? AND w2 = ? AND w3 = ?", (id1, id2, id3))
        row = self.cursor.fetchone()
        if row:
            return (math.log(row[0]) - TOTAL_CORPUS_LOG_PROB, "Trigram")
            
        score, eval_type = self.get_bigram_log_prob(w2, w3)
        return (BACKOFF_PENALTY + score, eval_type)

def run_interactive_test():
    db_path = os.path.join('android', 'app', 'src', 'main', 'assets', 'ngrams.db')
    print(f"Connecting to database: {db_path}...")
    tester = NGramTester(db_path)
    print("Connected successfully!\n")
    print("--------------------------------------------------")
    print("  AIRSPEECH N-GRAM PREDICTION TESTER")
    print("  Type 'quit' or 'exit' as context to stop.")
    print("--------------------------------------------------\n")

    while True:
        try:
            context = input("Enter previous sentence history (e.g. 'what is'): ").strip()
            if context.lower() in ('quit', 'exit'):
                break
                
            candidates_raw = input("Enter a comma-separated list of next word possibilities (e.g. 'up, playing, a'): ").strip()
            
            # Parse context to get W_minus_2 and W_minus_1
            tokens = context.split()
            w_minus_2 = tokens[-2] if len(tokens) >= 2 else ""
            w_minus_1 = tokens[-1] if len(tokens) >= 1 else ""
            
            candidates = [c.strip() for c in candidates_raw.split(',') if c.strip()]
            
            results = []
            
            for candidate in candidates:
                if w_minus_2 and w_minus_1:
                    score, eval_type = tester.get_trigram_log_prob(w_minus_2, w_minus_1, candidate)
                elif w_minus_1:
                    score, eval_type = tester.get_bigram_log_prob(w_minus_1, candidate)
                else:
                    score, eval_type = tester.get_unigram_log_prob(candidate)
                    
                results.append({"word": candidate, "score": score, "type": eval_type})
                
            # Sort highest score first
            results.sort(key=lambda x: x["score"], reverse=True)
            
            print("\n----- PREDICTION RANKING (~Log Prob Score) -----")
            # The score is a mathematically calculated log-probability. Closer to 0 (or positive) is better. Highly negative is bad.
            for i, res in enumerate(results):
                print(f" {i+1}. '{res['word']}'  | Score: {res['score']:.4f} | (Evaluated via {res['type']} Backoff)")
            print("------------------------------------------------\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error parsing input: {e}")

if __name__ == "__main__":
    run_interactive_test()
