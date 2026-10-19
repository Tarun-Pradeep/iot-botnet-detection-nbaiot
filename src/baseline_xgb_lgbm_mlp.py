"""
🎯 REMAINING MODELS - COMPLETE WITH ALL JOURNAL VISUALIZATIONS
XGBoost, LightGBM, MLP

Runs through ALL folds, creates ALL required visualizations for each model

python rest_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, brier_score_loss, log_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("⚠️  LightGBM not available")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("NBAIOT_BASE", ".")
DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")
OUTPUT_BASE = os.path.join(BASE_DIR, "results")

RANDOM_STATE = 42
N_FEATURES = 30
MAX_TRAIN = 1_000_000

print("="*80)
print("🎯 REMAINING MODELS - COMPLETE PIPELINE")
print("="*80)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_fold(fold_path):
    data = np.load(fold_path, allow_pickle=True)
    return {
        'X_train': data['X_train'],
        'X_test': data['X_test'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'device': str(data.get('test_device', 'Unknown')),
        'n_classes': int(data.get('n_classes', 11))
    }

def get_class_names(n_classes=11):
    return ['Benign', 
            'Mirai_ACK', 'Mirai_SYN', 'Mirai_UDP', 'Mirai_UDPPlain', 'Mirai_Scan',
            'Gafgyt_Combo', 'Gafgyt_Junk', 'Gafgyt_Scan', 'Gafgyt_TCP', 'Gafgyt_UDP'][:n_classes]

# ============================================================
# VISUALIZATION FUNCTIONS (REUSABLE FOR ALL MODELS)
# ============================================================

def plot_accuracy_per_fold(fold_metrics, model_name, save_path):
    print(f"  📊 Creating accuracy per fold plot for {model_name}...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    folds = [m['fold'] for m in fold_metrics]
    accuracies = [m['accuracy'] for m in fold_metrics]
    
    ax.plot(folds, accuracies, marker='o', linewidth=3, markersize=10, 
            color='#2ecc71', label='Fold Accuracy')
    ax.fill_between(folds, accuracies, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Accuracy per Fold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    mean_acc = np.mean(accuracies)
    ax.axhline(mean_acc, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_acc:.4f}')
    ax.legend(fontsize=11)
    
    for fold, acc in zip(folds, accuracies):
        ax.text(fold, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    print(f"  📊 Creating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1)
    axes[0].set_title('Raw Counts', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontweight='bold')
    axes[0].set_ylabel('True', fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, linewidths=1, vmin=0, vmax=1)
    axes[1].set_title('Normalized', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontweight='bold')
    axes[1].set_ylabel('True', fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_classification_report(y_true, y_pred, class_names, model_name, save_path):
    print(f"  📊 Creating classification report for {model_name}...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    precision = [report[c]['precision'] for c in class_names]
    recall = [report[c]['recall'] for c in class_names]
    f1 = [report[c]['f1-score'] for c in class_names]
    support = [report[c]['support'] for c in class_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Precision
    bars = axes[0, 0].barh(class_names, precision, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0, 0].set_xlabel('Precision', fontweight='bold')
    axes[0, 0].set_title('Precision per Class', fontweight='bold')
    axes[0, 0].set_xlim([0, 1.05])
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Recall
    bars = axes[0, 1].barh(class_names, recall, color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[0, 1].set_xlabel('Recall', fontweight='bold')
    axes[0, 1].set_title('Recall per Class', fontweight='bold')
    axes[0, 1].set_xlim([0, 1.05])
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # F1
    bars = axes[1, 0].barh(class_names, f1, color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
    axes[1, 0].set_title('F1-Score per Class', fontweight='bold')
    axes[1, 0].set_xlim([0, 1.05])
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Support
    bars = axes[1, 1].barh(class_names, support, color='#f39c12', alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Samples', fontweight='bold')
    axes[1, 1].set_title('Support per Class', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'{model_name} - Classification Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Text report
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write(f"{model_name} - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print(f"    ✅ Saved: {save_path}")

def plot_roc_curves(y_true, y_proba, class_names, model_name, save_path):
    print(f"  📊 Creating ROC curves for {model_name}...")
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        if i < y_proba.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, lw=2, label=f'{class_name} ({roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('FPR', fontweight='bold')
    ax1.set_ylabel('TPR', fontweight='bold')
    ax1.set_title('ROC - All Classes', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_proba[:, i])[0] 
                                        for i in range(min(n_classes, y_proba.shape[1]))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(min(n_classes, y_proba.shape[1])):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= min(n_classes, y_proba.shape[1])
    
    macro_auc = auc(all_fpr, mean_tpr)
    ax2.plot(all_fpr, mean_tpr, color='navy', lw=3, label=f'Macro-avg (AUC={macro_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlabel('FPR', fontweight='bold')
    ax2.set_ylabel('TPR', fontweight='bold')
    ax2.set_title('Macro-Average ROC', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'{model_name} - ROC Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_calibration_curves(y_true, y_proba, class_names, model_name, save_path):
    print(f"  📊 Creating calibration curves for {model_name}...")
    
    n_classes = len(class_names)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for i, class_name in enumerate(class_names):
        if i >= y_proba.shape[1]:
            axes[i].axis('off')
            continue
            
        ax = axes[i]
        y_true_binary = (y_true == i).astype(int)
        y_proba_class = y_proba[:, i]
        
        try:
            frac_pos, mean_pred = calibration_curve(y_true_binary, y_proba_class, 
                                                    n_bins=10, strategy='uniform')
            ax.plot(mean_pred, frac_pos, 'o-', linewidth=2, markersize=8, label=model_name)
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            brier = brier_score_loss(y_true_binary, y_proba_class)
            ax.set_title(f'{class_name}\nBrier: {brier:.4f}', fontweight='bold', fontsize=11)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Mean Predicted', fontweight='bold')
            ax.set_ylabel('Fraction Positive', fontweight='bold')
        except:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} - Calibration Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

# ============================================================
# MODEL TRAINING FUNCTIONS
# ============================================================

def train_model(model, model_name, output_dir, viz_dir):
    """Train a single model through all folds"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    folds = sorted(Path(DATA_DIR).glob("fold_*.npz"))
    
    if not folds:
        print(f"❌ No folds found in {DATA_DIR}")
        return None
    
    print(f"✅ Found {len(folds)} folds\n")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    
    start_time = time.time()
    
    for fold_idx, fold_path in enumerate(folds, 1):
        print(f"  Fold {fold_idx}/{len(folds)}: {fold_path.name}", end=" ... ")
        
        try:
            fold_data = load_fold(fold_path)
            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']
            
            # Feature selection
            selector = SelectKBest(f_classif, k=N_FEATURES)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
            # Subsample
            if len(X_train) > MAX_TRAIN:
                np.random.seed(RANDOM_STATE)
                idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
            
            # Class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
            
            fold_start = time.time()
            
            # Train based on model type
            if model_name == "MLP":
                model.fit(X_train, y_train)
            else:
                sample_weight = np.array([class_weights[y] for y in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weight)
            
            fold_time = time.time() - fold_start
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            fold_metrics.append({
                'fold': fold_idx,
                'accuracy': accuracy,
                'training_time': fold_time
            })
            
            print(f"✅ Acc: {accuracy:.4f} ({fold_time:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Combine results
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n{model_name} FINAL METRICS:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time: {total_time:.2f}s")
    
    # Get class names
    class_names = get_class_names(fold_data['n_classes'])
    
    # Generate ALL visualizations
    print(f"\n  Generating visualizations...")
    
    plot_accuracy_per_fold(fold_metrics, model_name, 
                           os.path.join(viz_dir, '1_accuracy_per_fold.png'))
    plot_confusion_matrix(y_true, y_pred, class_names, model_name,
                         os.path.join(viz_dir, '2_confusion_matrix.png'))
    plot_classification_report(y_true, y_pred, class_names, model_name,
                              os.path.join(viz_dir, '3_classification_report.png'))
    plot_roc_curves(y_true, y_proba, class_names, model_name,
                   os.path.join(viz_dir, '4_roc_curves.png'))
    plot_calibration_curves(y_true, y_proba, class_names, model_name,
                           os.path.join(viz_dir, '5_calibration_curves.png'))
    
    # Save results
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(output_dir, 'per_fold_results.csv'), index=False)
    
    final_metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_time': total_time
    }
    
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(output_dir, 'final_metrics.csv'), index=False)
    
    print(f"  ✅ Results saved to: {output_dir}")
    
    return final_metrics

# ============================================================
# MAIN
# ============================================================

def main():
    
    print("\n" + "="*80)
    print("STARTING TRAINING FOR ALL REMAINING MODELS")
    print("="*80)
    
    all_results = {}
    
    # XGBoost
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        
        output_dir = os.path.join(OUTPUT_BASE, "results_xgboost")
        viz_dir = os.path.join(output_dir, "visualizations")
        
        result = train_model(model, "XGBoost", output_dir, viz_dir)
        if result:
            all_results['XGBoost'] = result
    
    # LightGBM
    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        
        output_dir = os.path.join(OUTPUT_BASE, "results_lightgbm")
        viz_dir = os.path.join(output_dir, "visualizations")
        
        result = train_model(model, "LightGBM", output_dir, viz_dir)
        if result:
            all_results['LightGBM'] = result
    
    # MLP
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=50,
        early_stopping=True,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    output_dir = os.path.join(OUTPUT_BASE, "results_mlp")
    viz_dir = os.path.join(output_dir, "visualizations")
    
    result = train_model(model, "MLP", output_dir, viz_dir)
    if result:
        all_results['MLP'] = result
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - ALL MODELS")
    print(f"{'='*80}\n")
    
    comparison_df = pd.DataFrame(all_results).T
    print(comparison_df.to_string())
    
    comparison_df.to_csv(os.path.join(OUTPUT_BASE, "all_models_comparison.csv"))
    
    print(f"\n✅ ALL MODELS COMPLETE!")
    print(f"📂 Results saved to: {OUTPUT_BASE}")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()