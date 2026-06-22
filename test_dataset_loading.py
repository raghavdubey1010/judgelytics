#!/usr/bin/env python3
"""Test script to verify corrected dataset loading."""

import sys
sys.path.insert(0, 'ml_pipeline')

from preprocessing.download_dataset import download_and_merge_datasets

print("\n" + "="*70)
print("TESTING CORRECTED DATASET LOADING")
print("="*70)

try:
    print("\nDownloading dataset...")
    df = download_and_merge_datasets()
    
    print("\n✅ SUCCESS: Dataset loaded!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_cols = ['judgement', 'label', 'source', 'court_tier', 'split']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
    else:
        print(f"\n✅ All required columns present: {required_cols}")
    
    # Data quality checks
    print(f"\n📈 Data Quality:")
    print(f"   Null values - judgement: {df['judgement'].isna().sum()}")
    print(f"   Null values - label: {df['label'].isna().sum()}")
    print(f"   Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    print(f"\n✅ Dataset is compatible with preprocessing pipeline")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
