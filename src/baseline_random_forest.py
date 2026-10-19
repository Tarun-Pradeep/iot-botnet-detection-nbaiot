"""
🌲 RANDOM FOREST - COMPLETE WITH ALL JOURNAL VISUALIZATIONS
Runs through ALL folds, creates ALL required visualizations

Required visualizations:
✅ Accuracy per fold
✅ Confusion Matrix
✅ Classification Report
✅ ROC Curves
✅ Calibration Curves
+ Feature Importance & Additional visualizations

python model_rf.py
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("NBAIOT_BASE", ".")
DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results/results_random_forest")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FEATURES = 30
MAX_TRAIN = 1_000_000

print("="*80)
print("🌲 RANDOM FOREST - COMPLETE PIPELINE")
print("="*80)
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Visualization Directory: {VIZ_DIR}")
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
# VISUALIZATION FUNCTIONS (Similar to LR but customized for RF)
# ============================================================

def plot_accuracy_per_fold(fold_metrics, save_path):
    """Plot accuracy progression across folds"""
    print("  📊 Creating accuracy per fold plot...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    folds = [m['fold'] for m in fold_metrics]
    accuracies = [m['accuracy'] for m in fold_metrics]
    
    ax.plot(folds, accuracies, marker='o', linewidth=3, markersize=10, 
            color='#2ecc71', label='Fold Accuracy')
    ax.fill_between(folds, accuracies, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest - Accuracy per Fold', fontsize=14, fontweight='bold')
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

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    print("  📊 Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1)
    axes[0].set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontweight='bold')
    axes[0].set_ylabel('True Class', fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, linewidths=1, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix - Normalized', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Class', fontweight='bold')
    axes[1].set_ylabel('True Class', fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Random Forest - Confusion Matrix Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_classification_report(y_true, y_pred, class_names, save_path):
    print("  📊 Creating classification report visualization...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Precision
    ax = axes[0, 0]
    bars = ax.barh(classes, precision, color='#3498db', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Precision', fontweight='bold')
    ax.set_title('Precision per Class', fontweight='bold', fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, precision):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # Recall
    ax = axes[0, 1]
    bars = ax.barh(classes, recall, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_title('Recall per Class', fontweight='bold', fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, recall):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # F1-Score
    ax = axes[1, 0]
    bars = ax.barh(classes, f1, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_xlabel('F1-Score', fontweight='bold')
    ax.set_title('F1-Score per Class', fontweight='bold', fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, f1):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # Support
    ax = axes[1, 1]
    bars = ax.barh(classes, support, color='#f39c12', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Samples', fontweight='bold')
    ax.set_title('Support per Class', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, support):
        ax.text(val + max(support)*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(val)}', va='center', fontsize=9)
    
    plt.suptitle('Random Forest - Classification Report', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")
    
    # Save text report
    text_report_path = save_path.replace('.png', '.txt')
    with open(text_report_path, 'w') as f:
        f.write("RANDOM FOREST - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"    ✅ Saved text report: {text_report_path}")

def plot_roc_curves(y_true, y_proba, class_names, save_path):
    print("  📊 Creating ROC curves...")
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        if i < y_proba.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{class_name} (AUC={roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_title('ROC Curves - All Classes', fontweight='bold', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)
    
    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_proba[:, i])[0] 
                                        for i in range(min(n_classes, y_proba.shape[1]))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(min(n_classes, y_proba.shape[1])):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= min(n_classes, y_proba.shape[1])
    
    macro_auc = auc(all_fpr, mean_tpr)
    ax2.plot(all_fpr, mean_tpr, color='navy', lw=3,
            label=f'Macro-average ROC (AUC={macro_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax2.set_title('Macro-Average ROC Curve', fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Random Forest - ROC Curve Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_calibration_curves(y_true, y_proba, class_names, save_path):
    print("  📊 Creating calibration curves...")
    
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
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_proba_class, n_bins=10, strategy='uniform'
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                   linewidth=2, markersize=8, label='Random Forest')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            brier = brier_score_loss(y_true_binary, y_proba_class)
            
            ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontweight='bold')
            ax.set_title(f'{class_name}\nBrier Score: {brier:.4f}', 
                        fontweight='bold', fontsize=11)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        except:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Random Forest - Calibration Curves (Reliability Diagrams)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_feature_importance(model, save_path):
    """Plot feature importance - RF specific"""
    print("  📊 Creating feature importance plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Bar plot
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    bars = ax.barh(range(len(indices)), importances[indices], color=colors, 
                   alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Feature {i}' for i in indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
    ax.set_title('Top 30 Feature Importances', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    for bar, val in zip(bars, importances[indices]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=8)
    
    # Cumulative importance
    ax = axes[1]
    cumsum = np.cumsum(importances[indices])
    ax.plot(range(len(indices)), cumsum, marker='o', color='#e74c3c', 
            linewidth=3, markersize=6)
    ax.fill_between(range(len(indices)), cumsum, alpha=0.3, color='#e74c3c')
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, 
              label='80% threshold')
    ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=2,
              label='90% threshold')
    ax.set_xlabel('Number of Features', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Importance', fontweight='bold', fontsize=12)
    ax.set_title('Cumulative Feature Importance', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.suptitle('Random Forest - Feature Importance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_tree_depth_analysis(model, save_path):
    """Plot tree depth distribution - RF specific"""
    print("  📊 Creating tree depth analysis...")
    
    depths = [tree.get_depth() for tree in model.estimators_]
    n_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Tree depth histogram
    ax1.hist(depths, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(depths), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(depths):.1f}')
    ax1.set_xlabel('Tree Depth', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax1.set_title(f'Tree Depth Distribution (n_trees={len(depths)})', 
                 fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Leaves histogram
    ax2.hist(n_leaves, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(n_leaves), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(n_leaves):.1f}')
    ax2.set_xlabel('Number of Leaves', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax2.set_title('Leaf Node Distribution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Random Forest - Tree Structure Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_random_forest():
    """Main training function"""
    
    print("\n" + "="*80)
    print("STARTING RANDOM FOREST TRAINING")
    print("="*80 + "\n")
    
    folds = sorted(Path(DATA_DIR).glob("fold_*.npz"))
    
    if not folds:
        print(f"❌ ERROR: No fold files found in {DATA_DIR}")
        return
    
    print(f"✅ Found {len(folds)} folds\n")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    last_model = None  # Store last model for feature importance
    
    start_time = time.time()
    
    for fold_idx, fold_path in enumerate(folds, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold_idx}/{len(folds)}: {fold_path.name}")
        print(f"{'='*80}")
        
        try:
            fold_data = load_fold(fold_path)
            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']
            device = fold_data['device']
            n_classes = fold_data['n_classes']
            
            print(f"  Device: {device}")
            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
            
            # Feature selection
            print(f"\n  ⚙️  Preprocessing...")
            selector = SelectKBest(f_classif, k=N_FEATURES)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            print(f"     ✅ Features: {X_train.shape[1]}")
            
            # Subsample
            if len(X_train) > MAX_TRAIN:
                print(f"     📉 Subsampling: {len(X_train):,} → {MAX_TRAIN:,}")
                np.random.seed(RANDOM_STATE)
                idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
            
            # Class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
            
            # Train
            print(f"\n  🚀 Training Random Forest...")
            fold_start = time.time()
            
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_leaf=20,
                class_weight=class_weights,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(X_train, y_train)
            fold_time = time.time() - fold_start
            last_model = model  # Save for feature importance
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"     ✅ Training completed in {fold_time:.2f}s")
            print(f"     📊 Accuracy: {accuracy:.4f}")
            
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            fold_metrics.append({
                'fold': fold_idx,
                'device': device,
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'training_time': fold_time
            })
            
        except Exception as e:
            print(f"\n  ❌ ERROR in fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Combine results
    print(f"\n{'='*80}")
    print("COMBINING RESULTS FROM ALL FOLDS")
    print(f"{'='*80}\n")
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("FINAL PERFORMANCE METRICS")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*80)
    
    class_names = get_class_names(n_classes)
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING ALL VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    plot_accuracy_per_fold(fold_metrics, 
                           os.path.join(VIZ_DIR, '1_accuracy_per_fold.png'))
    plot_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(VIZ_DIR, '2_confusion_matrix.png'))
    plot_classification_report(y_true, y_pred, class_names,
                              os.path.join(VIZ_DIR, '3_classification_report.png'))
    plot_roc_curves(y_true, y_proba, class_names,
                   os.path.join(VIZ_DIR, '4_roc_curves.png'))
    plot_calibration_curves(y_true, y_proba, class_names,
                           os.path.join(VIZ_DIR, '5_calibration_curves.png'))
    
    if last_model:
        plot_feature_importance(last_model,
                               os.path.join(VIZ_DIR, '6_feature_importance.png'))
        plot_tree_depth_analysis(last_model,
                                os.path.join(VIZ_DIR, '7_tree_depth_analysis.png'))
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, 'per_fold_results.csv'), index=False)
    
    final_metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_time_sec': total_time,
        'total_samples': len(y_true)
    }
    
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, 'final_metrics.csv'), index=False)
    
    print(f"\n{'='*80}")
    print("✅ RANDOM FOREST - COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📂 Outputs: {OUTPUT_DIR}")
    print(f"📊 Visualizations: {VIZ_DIR}\n")

if __name__ == "__main__":
    try:
        train_random_forest()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
