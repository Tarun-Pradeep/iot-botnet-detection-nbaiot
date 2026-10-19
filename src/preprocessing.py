"""
═══════════════════════════════════════════════════════════════
STEP 1: PREPROCESSING - GUARANTEED CROSS-DEVICE SPLIT
═══════════════════════════════════════════════════════════════
This version FORCES proper cross-device evaluation with verification.
Expected F1 scores: 75-90%
═══════════════════════════════════════════════════════════════
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_ROOT = os.environ.get("NBAIOT_DATA", "data")
OUTPUT_DIR = os.environ.get("NBAIOT_PREPROCESSED", "preprocessed_data")
RANDOM_STATE = 42

# IMPORTANT: Subsample training data to make it harder (more realistic)
TRAIN_SUBSAMPLE = 0.30  # Use only 30% of training data per fold

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("═" * 70)
print("  STEP 1: CROSS-DEVICE PREPROCESSING")
print("  (With 30% training data subsampling for realistic difficulty)")
print("═" * 70)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_nbaiot_data(data_root: str) -> pd.DataFrame:
    """Load all N-BaIoT CSVs."""
    print(f"\n📂 Loading data from: {data_root}")
    frames = []
    
    for device_dir in sorted(Path(data_root).iterdir()):
        if not device_dir.is_dir():
            continue
        
        device_name = device_dir.name.replace('_', ' ').strip()
        
        # Benign
        benign_csv = device_dir / "benign_traffic.csv"
        if benign_csv.exists():
            df = pd.read_csv(benign_csv, header=0)
            df["label"] = "benign"
            df["device"] = device_name
            frames.append(df)
            print(f"   ✓ {device_name}/benign ({len(df):,})")
        
        # Attacks
        for attack_folder in ["gafgyt_attacks", "mirai_attacks"]:
            attack_dir = device_dir / attack_folder
            if not attack_dir.exists():
                continue
            prefix = attack_folder.split("_")[0]
            
            for csv_file in sorted(attack_dir.glob("*.csv")):
                df = pd.read_csv(csv_file, header=0)
                df["label"] = f"{prefix}_{csv_file.stem}"
                df["device"] = device_name
                frames.append(df)
    
    df_all = pd.concat(frames, ignore_index=True)
    
    print(f"\n Total: {df_all.shape[0]:,} samples, {df_all.shape[1]-2} features")
    print(f"   Devices: {df_all['device'].nunique()}")
    print(f"   Unique devices: {sorted(df_all['device'].unique())}")
    
    return df_all


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING WITH STRICT VERIFICATION
# ═══════════════════════════════════════════════════════════════
def preprocess_fold(df: pd.DataFrame, test_device: str, fold_num: int):
    """Create one cross-device fold with STRICT verification."""
    print(f"\n{'─'*70}")
    print(f"  FOLD {fold_num}: Test Device = {test_device}")
    print(f"{'─'*70}")
    
    feat_cols = [c for c in df.columns if c not in ("label", "device")]
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    
    # STRICT DEVICE SPLIT
    train_df = df_clean[df_clean["device"] != test_device].copy()
    test_df  = df_clean[df_clean["device"] == test_device].copy()
    
    # ═══ VERIFICATION ═══
    train_devices = set(train_df['device'].unique())
    test_devices = set(test_df['device'].unique())
    
    if test_device not in test_devices:
        print(f"   ❌ FATAL: Test device not in test set!")
        return None
    
    if len(train_devices & test_devices) > 0:
        print(f"   ❌ FATAL: Device overlap detected!")
        print(f"      Overlap: {train_devices & test_devices}")
        return None
    
    print(f"   ✅ Cross-device verified")
    print(f"      Train: {sorted(train_devices)}")
    print(f"      Test:  {test_device}")
    print(f"      Train samples (before subsample): {len(train_df):,}")
    
    # ═══ SUBSAMPLE TRAINING DATA (makes it harder) ═══
    if TRAIN_SUBSAMPLE < 1.0:
        train_df, _ = train_test_split(
            train_df,
            train_size=TRAIN_SUBSAMPLE,
            stratify=train_df["label"],
            random_state=RANDOM_STATE
        )
        print(f"      Train samples (after {TRAIN_SUBSAMPLE*100:.0f}% subsample): {len(train_df):,}")
    
    print(f"      Test samples: {len(test_df):,}")
    
    # Extract features
    X_train_full = train_df[feat_cols].values.astype(np.float32)
    X_test       = test_df[feat_cols].values.astype(np.float32)
    
    # Labels
    le = LabelEncoder()
    le.fit(df_clean["label"])
    y_train_full = le.transform(train_df["label"]).astype(np.int64)
    y_test       = le.transform(test_df["label"]).astype(np.int64)
    
    # Z-score (fit on train only!)
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
    X_test       = scaler.transform(X_test).astype(np.float32)
    
    # Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )
    
    print(f"   ✓ Final split: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
    
    # Save
    fold_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'test_device': test_device,
        'n_classes': len(le.classes_),
        'class_names': le.classes_.tolist(),
        'feature_names': feat_cols,
    }
    
    filename = f"{OUTPUT_DIR}/fold_{fold_num:02d}_{test_device.replace(' ', '_')}.npz"
    np.savez_compressed(filename, **fold_data)
    print(f"   💾 Saved: {filename}")
    
    return fold_data


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    df = load_nbaiot_data(DATA_ROOT)
    unique_devices = sorted(df["device"].unique())
    
    print(f"\n{'═'*70}")
    print(f"  Creating {len(unique_devices)} cross-device folds...")
    print(f"  Training data: {TRAIN_SUBSAMPLE*100:.0f}% per fold (subsampled)")
    print(f"{'═'*70}")
    
    fold_info = []
    for i, test_device in enumerate(unique_devices, 1):
        result = preprocess_fold(df, test_device, i)
        if result:
            fold_info.append({
                'fold_num': i,
                'test_device': test_device,
                'n_train': len(result['X_train']),
                'n_test': len(result['X_test']),
            })
    
    summary_df = pd.DataFrame(fold_info)
    summary_df.to_csv(f"{OUTPUT_DIR}/fold_summary.csv", index=False)
    
    print(f"\n{'═'*70}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  ✅ {len(fold_info)} folds created")
    print(f"  ✅ Files: {OUTPUT_DIR}/")
    print(f"\n  Next: Run step2_train_models.py")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()