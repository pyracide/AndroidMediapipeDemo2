import sqlite3
import os
import re
from collections import Counter
import sys

# Replace this with the exact name of your downloaded .txt file
DATASET_FILENAME = "subtitle_dataset.txt"

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()

def create_db(db_path, txt_path):
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    # Optimize SQLite for massive inserts
    conn.execute('PRAGMA journal_mode = OFF')
    conn.execute('PRAGMA synchronous = 0')
    conn.execute('PRAGMA cache_size = 1000000')
    conn.execute('PRAGMA locking_mode = EXCLUSIVE')
    
    cursor = conn.cursor()
    
    print("Creating relational tables...")
    # Relational ID mapping to save 99% of file space and RAM
    cursor.execute('CREATE TABLE dictionary (id INTEGER PRIMARY KEY, word TEXT UNIQUE)')
    cursor.execute('CREATE TABLE unigrams (w1 INTEGER PRIMARY KEY, count INTEGER)')
    cursor.execute('CREATE TABLE bigrams (w1 INTEGER, w2 INTEGER, count INTEGER, PRIMARY KEY(w1, w2))')
    cursor.execute('CREATE TABLE trigrams (w1 INTEGER, w2 INTEGER, w3 INTEGER, count INTEGER, PRIMARY KEY(w1, w2, w3))')
    
    # Store dynamic IDs
    word_to_id = {}
    next_id = 1
    
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    
    print(f"Reading dataset: {txt_path} (This might take a while...)")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                tokens = tokenize(line)
                if not tokens: continue
                
                # Convert string token to integer ID dynamically
                token_ids = []
                for t in tokens:
                    if t not in word_to_id:
                        word_to_id[t] = next_id
                        next_id += 1
                    token_ids.append(word_to_id[t])
                
                unigrams.update(token_ids)
                if len(token_ids) >= 2:
                    bigrams.update(zip(token_ids, token_ids[1:]))
                if len(token_ids) >= 3:
                    trigrams.update(zip(token_ids, token_ids[1:], token_ids[2:]))
                
                if i % 100000 == 0 and i > 0:
                    print(f"Processed {i} lines...")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{txt_path}'. Ensure you placed the downloaded text file in this directory.")
        sys.exit(1)
        
    print("Writing to SQLite database...")
    
    # Prune rare sequences to save phone space
    print("Sorting and capping sequences to Top-K to ensure a sub-50MB mobile file size...")
    
    # Cap dimensions mathematically
    pruned_uni = dict(unigrams.most_common(50000))
    pruned_bi = dict(bigrams.most_common(800000))  # Massively increased Bigrams for stronger Backoff
    pruned_tri = dict(trigrams.most_common(600000))
    
    # Free up system RAM now before DB commit
    unigrams.clear()
    bigrams.clear()
    trigrams.clear()
    
    # Filter dictionary to ONLY save text strings that survived the Top-K cut
    valid_ids = set()
    for k in pruned_uni.keys():
        valid_ids.add(k)
    for k in pruned_bi.keys():
        valid_ids.update(k)
    for k in pruned_tri.keys():
        valid_ids.update(k)
        
    pruned_dict = {w: i for w, i in word_to_id.items() if i in valid_ids}
    word_to_id.clear()
    
    # Insert dictionary mappings
    cursor.executemany('INSERT INTO dictionary (word, id) VALUES (?, ?)',
                       [(k, v) for k, v in pruned_dict.items()])
                       
    cursor.executemany('INSERT INTO unigrams (w1, count) VALUES (?, ?)',
                       [(k, v) for k, v in pruned_uni.items()])
                       
    cursor.executemany('INSERT INTO bigrams (w1, w2, count) VALUES (?, ?, ?)',
                       [(k[0], k[1], v) for k, v in pruned_bi.items()])
                       
    cursor.executemany('INSERT INTO trigrams (w1, w2, w3, count) VALUES (?, ?, ?, ?)',
                       [(k[0], k[1], k[2], v) for k, v in pruned_tri.items()])
                       
    conn.commit()
    conn.close()
    print(f"Successfully generated optimized Relational N-gram database at {db_path}")

if __name__ == '__main__':
    # Save into the Android project assets folder
    assets_dir = os.path.join('android', 'app', 'src', 'main', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    db_path = os.path.join(assets_dir, 'ngrams.db')
    
    # Assume the text file is placed in the project root
    txt_path = os.path.join(os.getcwd(), DATASET_FILENAME)
    
    create_db(db_path, txt_path)
