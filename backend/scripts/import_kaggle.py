"""
Kaggle Dataset Importer for Judgelytics

This script allows you to import Supreme Court decisions from Kaggle datasets 
(e.g., "Indian Supreme Court Judgments") into the Judgelytics local database.

Usage:
  python import_kaggle.py --file path/to/dataset.csv

Requirements:
  pip install pandas
"""

import argparse
import json
import os
import re
import uuid
import logging
import sys

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required to run this script. Run `pip install pandas`")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'judgements.json')
CONSUMER_KEYWORDS = [
    "consumer protection", "deficiency of service", "unfair trade practice",
    "ncdrc", "district forum", "medical negligence", "builder delay", 
    "r.e.r.a", "rera", "insurance claim", "repudiation"
]

def load_existing_db():
    if not os.path.exists(DB_PATH):
        return {"judgements": []}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_consumer_case(text):
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CONSUMER_KEYWORDS)

def map_sector(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ["medical", "doctor", "hospital", "surgery"]): return "Healthcare"
    if any(k in text_lower for k in ["builder", "flat", "possession", "rera", "allottee"]): return "Real Estate"
    if any(k in text_lower for k in ["bank", "cheque", "loan", "upi", "credit card"]): return "Banking"
    if any(k in text_lower for k in ["insurance", "policy", "premium", "repudiation"]): return "Insurance"
    if any(k in text_lower for k in ["flight", "airline", "train", "railway", "ticket", "hotel"]): return "Travel"
    if any(k in text_lower for k in ["amazon", "flipkart", "ecommerce", "online", "delivery"]): return "E-commerce"
    if any(k in text_lower for k in ["telecom", "mobile", "sim", "vas", "broadband"]): return "Telecom"
    if any(k in text_lower for k in ["school", "college", "coaching", "fee", "student"]): return "Education"
    if any(k in text_lower for k in ["car", "vehicle", "tractor", "automobile"]): return "Automobile"
    return "General Consumer"

def process_kaggle_csv(filepath):
    logging.info(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return

    # Assuming standard Kaggle SC datasets have columns like: 'title', 'date', 'judgment_text'
    # We will try to dynamically find the relevant columns
    text_col = next((col for col in df.columns if col.lower() in ['text', 'judgment', 'judgment_text', 'summary', 'description']), None)
    title_col = next((col for col in df.columns if col.lower() in ['title', 'case_name', 'name', 'party']), None)
    date_col = next((col for col in df.columns if col.lower() in ['date', 'year', 'judgment_date']), None)

    if not text_col or not title_col:
        logging.error("Could not identify 'title' or 'text' columns in the dataset.")
        logging.info(f"Available columns: {list(df.columns)}")
        return

    logging.info("Filtering for consumer cases...")
    
    # Filter dataset based on keywords
    consumer_cases = df[df[text_col].apply(is_consumer_case)]
    
    if len(consumer_cases) == 0:
        logging.info("No consumer cases found in this dataset.")
        return

    logging.info(f"Found {len(consumer_cases)} consumer-related cases. Processing...")
    
    db_data = load_existing_db()
    existing_titles = {j['title'].lower() for j in db_data['judgements']}
    
    added_count = 0
    for _, row in consumer_cases.iterrows():
        title = str(row[title_col]).strip()
        if title.lower() in existing_titles:
            continue
            
        full_text = str(row[text_col]).strip()
        
        # Extract year
        year = 2000
        if date_col:
            date_val = str(row[date_col])
            match = re.search(r'\b(19|20)\d{2}\b', date_val)
            if match:
                year = int(match.group(0))
        
        # Generate a summary (first 250 chars)
        summary = full_text[:250] + "..." if len(full_text) > 250 else full_text
        
        sector = map_sector(full_text)
        
        # Determine outcome heuristically
        outcome = "Allowed" if "allowed" in full_text[-500:].lower() else "Dismissed"
        
        case_obj = {
            "id": f"k_{uuid.uuid4().hex[:8]}",
            "title": title,
            "court": "Supreme Court of India",
            "year": year,
            "citation": "Kaggle Dataset",
            "sector": sector,
            "outcome": outcome,
            "summary": summary,
            "key_principle": "Extracted via Automated Kaggle Script",
            "sections": ["Kaggle Import"],
            "tags": ["kaggle", "supreme court", sector.lower()]
        }
        
        db_data["judgements"].append(case_obj)
        existing_titles.add(title.lower())
        added_count += 1
        
        # Limit to adding max 100 cases per run to prevent DB bloat
        if added_count >= 100:
            logging.info("Reached maximum import limit of 100 cases per run.")
            break

    if added_count > 0:
        save_db(db_data)
        logging.info(f"Successfully imported {added_count} cases into Judgelytics database!")
    else:
        logging.info("No new cases were added (they might already exist).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import consumer cases from Kaggle Supreme Court CSV.")
    parser.add_argument("--file", required=True, help="Path to the Kaggle CSV dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logging.error(f"File not found: {args.file}")
        sys.exit(1)
        
    process_kaggle_csv(args.file)
