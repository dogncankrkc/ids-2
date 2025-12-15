"""
Dataset Generator Script for CIC-IoT-2023.

Purpose:
- Reads the massive raw CSV file in chunks to save RAM.
- Explicitly removes 'nan' (null) values.
- Maps detailed attack labels to 8 main categories (DDoS, DoS, Web, etc.).
- Collects exactly 50,000 samples per category (caps majority classes).
- Collects ALL samples for minority classes (Web, Backdoor, etc.) if they are < 50k.
- Saves a new, lightweight, and balanced CSV file for training.
"""

import pandas as pd
import os
import numpy as np

# ------------------------
# CONFIGURATION
# ------------------------
# Input file (The huge raw dataset)
INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"

# Output file (The clean, balanced dataset to be created)
OUTPUT_PATH = "data/processed/CIC2023_Balanced_50k.csv"

# Target samples per class (Cap limit)
SAMPLES_PER_CLASS = 50000 

def get_category(label):
    """
    Maps the raw label to one of the 8 main categories.
    Must match the logic in preprocess.py exactly.
    """
    # Convert to string, strip whitespace, and uppercase
    label = str(label).strip().upper()
    
    # Explicitly handle NaN or empty strings
    if label == 'NAN' or label == '':
        return 'DROP'
        
    if 'BENIGN' in label:
        return 'Benign'
    elif 'DDOS' in label:
        return 'DDoS'
    elif 'DOS' in label:
        return 'DoS'
    # Recon types
    elif 'RECON' in label or 'VULNERABILITYSCAN' in label or 'PING' in label or 'PORTSCAN' in label or 'OSSCAN' in label or 'HOSTDISCOVERY' in label:
        return 'Recon'
    # Web / Injection / Malware types (Grouping these is crucial for the 8-class logic)
    elif 'XSS' in label or 'SQL' in label or 'UPLOAD' in label or 'BROWSER' in label or 'COMMAND' in label or 'BACKDOOR' in label or 'MALWARE' in label:
        return 'Web'
    elif 'BRUTEFORCE' in label or 'DICTIONARY' in label:
        return 'BruteForce'
    elif 'SPOOFING' in label or 'MITM' in label:
        return 'Spoofing'
    elif 'MIRAI' in label:
        return 'Mirai'
    else:
        return 'Other'

def create_balanced_dataset():
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"[WARNING] Output file '{OUTPUT_PATH}' already exists. It will be overwritten.")

    # Dictionary to track how many samples we have collected for each category
    collected_counts = {
        'Benign': 0, 'DDoS': 0, 'DoS': 0, 'Recon': 0, 
        'Web': 0, 'BruteForce': 0, 'Spoofing': 0, 'Mirai': 0
    }
    
    sampled_chunks = []
    chunk_size = 1000000 # Read 1 million rows at a time
    
    print(f"[INFO] Processing {INPUT_PATH}...")
    print(f"[GOAL] Collect max {SAMPLES_PER_CLASS} samples per category (8 Classes).")
    print(f"[NOTE] Minority classes (Web, etc.) will be fully preserved.")

    try:
        # Iterate over the file in chunks
        for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=chunk_size)):
            print(f"   -> Processing Chunk {i+1}...")
            
            # Clean column names
            chunk.columns = chunk.columns.str.strip()
            
            # Find the label column
            if 'multiclass_label' in chunk.columns:
                label_col = 'multiclass_label'
            elif 'label' in chunk.columns:
                label_col = 'label'
            else:
                print(f"[ERROR] Label column not found in chunk {i+1}. Skipping.")
                continue

            # 1. DROP NAN VALUES explicitly
            chunk = chunk.dropna(subset=[label_col])
            
            # 2. MAP LABELS
            chunk['Mapped_Label'] = chunk[label_col].apply(get_category)
            
            # Remove 'DROP' (NaNs) or 'Other' (Unknowns)
            chunk = chunk[~chunk['Mapped_Label'].isin(['DROP', 'Other'])]
            
            # 3. SELECT SAMPLES (Sampling Logic)
            for category, group in chunk.groupby('Mapped_Label'):
                
                # How many do we already have?
                current_count = collected_counts.get(category, 0)
                
                # If we reached the limit (50k), skip this class
                if current_count >= SAMPLES_PER_CLASS:
                    continue
                
                # How many do we need?
                needed = SAMPLES_PER_CLASS - current_count
                
                # If chunk has more than needed, sample randomly.
                # If chunk has less than needed (Minority classes), take ALL.
                if len(group) > needed:
                    taken = group.sample(n=needed, random_state=42)
                else:
                    taken = group
                
                sampled_chunks.append(taken)
                collected_counts[category] += len(taken)
            
            # Early Exit: If all categories reached 50k, stop reading.
            if all(count >= SAMPLES_PER_CLASS for count in collected_counts.values()):
                print("[SUCCESS] All categories reached the target limit! Stopping early.")
                break
                
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        return

    print("[INFO] Concatenating selected data...")
    if not sampled_chunks:
        print("[ERROR] No data collected. Check label mapping or input file.")
        return

    final_df = pd.concat(sampled_chunks, ignore_index=True)
    
    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*40)
    print("FINAL DATASET DISTRIBUTION (ON DISK)")
    print("="*40)
    print(final_df['Mapped_Label'].value_counts())
    print("-" * 40)
    
    # Save to CSV
    print(f"\n[SAVING] Writing to {OUTPUT_PATH}...")
    final_df.to_csv(OUTPUT_PATH, index=False)
    print("[DONE] Clean dataset is ready.")

if __name__ == "__main__":
    create_balanced_dataset()