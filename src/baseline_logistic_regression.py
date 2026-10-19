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

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("NBAIOT_BASE", ".")
DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results/results_logistic_regression")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FEATURES = 30
MAX_TRAIN = 1_000_000

print("="*80)
print("🎯 LOGISTIC REGRESSION - COMPLETE PIPELINE")
print("="*80)
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Visualization Directory: {VIZ_DIR}")
print("="*80)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_fold(fold_path):
    """Load fold data"""
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
    """Get class names"""
    return ['Benign', 
            'Mirai_ACK', 'Mirai_SYN', 'Mirai_UDP', 'Mirai_UDPPlain', 'Mirai_Scan',
            'Gafgyt_Combo', 'Gafgyt_Junk', 'Gafgyt_Scan', 'Gafgyt_TCP', 'Gafgyt_UDP'][:n_classes]

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_accuracy_loss_per_fold(fold_metrics, save_path):
    """Plot accuracy and loss progression across folds"""
    print("  📊 Creating accuracy & loss per fold plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    folds = [m['fold'] for m in fold_metrics]
    accuracies = [m['accuracy'] for m in fold_metrics]
    losses = [m['loss'] for m in fold_metrics]
    
    # Accuracy plot
    ax1.plot(folds, accuracies, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax1.fill_between(folds, accuracies, alpha=0.3, color='#2ecc71')
    ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy per Fold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add mean line
    mean_acc = np.mean(accuracies)
    ax1.axhline(mean_acc, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_acc:.4f}')
    ax1.legend()
    
    # Add value labels
    for i, (fold, acc) in enumerate(zip(folds, accuracies)):
        ax1.text(fold, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
    
    # Loss plot
    ax2.plot(folds, losses, marker='s', linewidth=2, markersize=8, color='#e74c3c')
    ax2.fill_between(folds, losses, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Log Loss per Fold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_loss = np.mean(losses)
    ax2.axhline(mean_loss, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_loss:.4f}')
    ax2.legend()
    
    # Add value labels
    for i, (fold, loss) in enumerate(zip(folds, losses)):
        ax2.text(fold, loss + 0.02, f'{loss:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Logistic Regression - Performance Across Folds', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
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
    
    plt.suptitle('Logistic Regression - Confusion Matrix Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_classification_report(y_true, y_pred, class_names, save_path):
    """Visualize classification report"""
    print("  📊 Creating classification report visualization...")
    
    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    # Extract metrics
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
    
    plt.suptitle('Logistic Regression - Classification Report', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")
    
    # Save text report
    text_report_path = save_path.replace('.png', '.txt')
    with open(text_report_path, 'w') as f:
        f.write("LOGISTIC REGRESSION - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"    ✅ Saved text report: {text_report_path}")

def plot_roc_curves(y_true, y_proba, class_names, save_path):
    """Plot ROC curves for all classes"""
    print("  📊 Creating ROC curves...")
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Individual ROC curves
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
    
    plt.suptitle('Logistic Regression - ROC Curve Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_calibration_curves(y_true, y_proba, class_names, save_path):
    """Plot calibration curves"""
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
        
        # Get binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_proba_class = y_proba[:, i]
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_proba_class, n_bins=10, strategy='uniform'
            )
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                   linewidth=2, markersize=8, label='Logistic Regression')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            # Calculate Brier score
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
    
    # Hide extra subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Logistic Regression - Calibration Curves (Reliability Diagrams)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {save_path}")

def plot_additional_visualizations(y_true, y_pred, y_proba, class_names, metrics, save_dir):
    """Additional visualizations for journal"""
    print("  📊 Creating additional visualizations...")
    
    # 1. Overall metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                     metrics['recall'], metrics['f1']]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Overall Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
               f'{val:.4f}\n({val*100:.2f}%)', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_overall_metrics.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Class-wise performance heatmap
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    
    class_metrics = []
    for class_name in class_names:
        class_metrics.append([
            report[class_name]['precision'],
            report[class_name]['recall'],
            report[class_name]['f1-score']
        ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(class_metrics, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=class_names, cbar_kws={'label': 'Score'},
                linewidths=1, ax=ax)
    ax.set_title('Per-Class Performance Heatmap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_class_performance_heatmap.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✅ Saved additional visualizations")

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_logistic_regression():
    """Main training function"""
    
    print("\n" + "="*80)
    print("STARTING LOGISTIC REGRESSION TRAINING")
    print("="*80 + "\n")
    
    # Find folds
    folds = sorted(Path(DATA_DIR).glob("fold_*.npz"))
    
    if not folds:
        print(f"❌ ERROR: No fold files found in {DATA_DIR}")
        return
    
    print(f"✅ Found {len(folds)} folds")
    print(f"   Folds: {[f.name for f in folds]}\n")
    
    # Storage for results
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    
    start_time = time.time()
    
    # Train on each fold
    for fold_idx, fold_path in enumerate(folds, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold_idx}/{len(folds)}: {fold_path.name}")
        print(f"{'='*80}")
        
        try:
            # Load data
            fold_data = load_fold(fold_path)
            X_train = fold_data['X_train']
            X_test = fold_data['X_test']
            y_train = fold_data['y_train']
            y_test = fold_data['y_test']
            device = fold_data['device']
            n_classes = fold_data['n_classes']
            
            print(f"  Device: {device}")
            print(f"  Train samples: {len(X_train):,}")
            print(f"  Test samples: {len(X_test):,}")
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Classes: {n_classes}")
            
            # Feature scaling
            print(f"\n  ⚙️  Preprocessing...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Feature selection
            selector = SelectKBest(f_classif, k=N_FEATURES)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            print(f"     ✅ Features: {X_train.shape[1]}")
            
            # Subsample if needed
            if len(X_train) > MAX_TRAIN:
                print(f"     📉 Subsampling: {len(X_train):,} → {MAX_TRAIN:,}")
                np.random.seed(RANDOM_STATE)
                idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
            
            # Train model
            print(f"\n  🚀 Training Logistic Regression...")
            fold_start = time.time()
            
            model = LogisticRegression(
                max_iter=400,
                solver='saga',
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(X_train, y_train)
            fold_time = time.time() - fold_start
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)
            
            print(f"     ✅ Training completed in {fold_time:.2f}s")
            print(f"     📊 Accuracy: {accuracy:.4f}")
            print(f"     📊 Log Loss: {loss:.4f}")
            
            # Store results
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            fold_metrics.append({
                'fold': fold_idx,
                'device': device,
                'accuracy': accuracy,
                'loss': loss,
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
    
    # Combine all results
    print(f"\n{'='*80}")
    print("COMBINING RESULTS FROM ALL FOLDS")
    print(f"{'='*80}\n")
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)
    
    # Calculate final metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    final_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print("FINAL PERFORMANCE METRICS")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*80)
    
    # Get class names
    class_names = get_class_names(n_classes)
    
    # Create all visualizations
    print(f"\n{'='*80}")
    print("GENERATING ALL VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # 1. Accuracy & Loss per fold (REQUIRED)
    plot_accuracy_loss_per_fold(fold_metrics, 
                                os.path.join(VIZ_DIR, '1_accuracy_loss_per_fold.png'))
    
    # 2. Confusion Matrix (REQUIRED)
    plot_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(VIZ_DIR, '2_confusion_matrix.png'))
    
    # 3. Classification Report (REQUIRED)
    plot_classification_report(y_true, y_pred, class_names,
                              os.path.join(VIZ_DIR, '3_classification_report.png'))
    
    # 4. ROC Curves (REQUIRED)
    plot_roc_curves(y_true, y_proba, class_names,
                   os.path.join(VIZ_DIR, '4_roc_curves.png'))
    
    # 5. Calibration Curves (REQUIRED)
    plot_calibration_curves(y_true, y_proba, class_names,
                           os.path.join(VIZ_DIR, '5_calibration_curves.png'))
    
    # 6. Additional visualizations
    plot_additional_visualizations(y_true, y_pred, y_proba, class_names, 
                                  final_metrics, VIZ_DIR)
    
    # Save results to CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Per-fold results
    fold_df = pd.DataFrame(fold_metrics)
    fold_csv = os.path.join(OUTPUT_DIR, 'per_fold_results.csv')
    fold_df.to_csv(fold_csv, index=False)
    print(f"  ✅ Per-fold results: {fold_csv}")
    
    # Final metrics
    final_df = pd.DataFrame([final_metrics])
    final_df['model'] = 'Logistic Regression'
    final_df['total_time_sec'] = total_time
    final_df['total_samples'] = len(y_true)
    final_csv = os.path.join(OUTPUT_DIR, 'final_metrics.csv')
    final_df.to_csv(final_csv, index=False)
    print(f"  ✅ Final metrics: {final_csv}")
    
    # Summary
    summary_path = os.path.join(OUTPUT_DIR, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("LOGISTIC REGRESSION - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of folds: {len(folds)}\n")
        f.write(f"Total samples: {len(y_true):,}\n")
        f.write(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n\n")
        f.write("FINAL METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
        f.write("VISUALIZATIONS CREATED:\n")
        f.write("-"*40 + "\n")
        f.write("1. Accuracy & Loss per fold\n")
        f.write("2. Confusion Matrix\n")
        f.write("3. Classification Report\n")
        f.write("4. ROC Curves\n")
        f.write("5. Calibration Curves\n")
        f.write("6. Overall Metrics\n")
        f.write("7. Class Performance Heatmap\n")
    print(f"  ✅ Summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ LOGISTIC REGRESSION - COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📂 All outputs saved to: {OUTPUT_DIR}")
    print(f"📊 All visualizations saved to: {VIZ_DIR}")
    print(f"\n🎉 Training completed successfully!\n")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        train_logistic_regression()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
