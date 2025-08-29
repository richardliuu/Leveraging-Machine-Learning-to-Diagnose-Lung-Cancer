"""
Random Forest Seed Stability Analysis
Tests Random Forest performance across multiple random seeds (0-50)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataHandling:
    def __init__(self):
        self.data = None
        self.feature_cols = None
        self.groups = None
        self.X = None
        self.y = None

    def load_data(self, path="data/jitter_shimmerlog.csv"):
        """Load original data"""
        self.data = pd.read_csv(path)
        
        # Features: drop non-feature columns
        self.X = self.data.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 
                                         'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
        self.feature_cols = self.X.columns.tolist()
        
        self.y = (self.data['cancer_stage'] > 0).astype(int)
        
        self.groups = self.data['patient_id']
        
        print(f"Data loaded: {len(self.X)} samples, {len(self.X.columns)} features")
        print(f"Class distribution: {self.y.value_counts().to_dict()}")

def train_rf_single_seed(handler, seed, verbose=False):
    """Train and evaluate Random Forest with a specific seed"""
    
    # Set all random seeds
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 4-fold cross-validation
    gkf = StratifiedGroupKFold(n_splits=4)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups), 1):
        # Split data
        X_train, X_test = handler.X.iloc[train_idx], handler.X.iloc[test_idx]
        y_train, y_test = handler.y.iloc[train_idx], handler.y.iloc[test_idx]
        
        # Train Random Forest with current seed
        rf = RandomForestClassifier(
            criterion="log_loss",
            n_estimators=200,
            max_depth=5,
            max_features=0.6,
            min_samples_split=25,
            min_samples_leaf=10,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
            random_state=seed,  
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'oob_score': rf.oob_score_ if rf.oob_score else None
        }
        
        fold_results.append(fold_metrics)
        
        if verbose and fold == 1:
            print(f"Seed {seed}, Fold 1: AUC={fold_metrics['auc']:.4f}, Acc={fold_metrics['accuracy']:.4f}")
    
    return fold_results

def multi_seed_rf_analysis(handler, seed_range=(0, 51)):
    """Run Random Forest with multiple seeds and analyze stability"""
    
    start_seed, end_seed = seed_range
    n_seeds = end_seed - start_seed
    print(f"\n=== Testing Random Forest with Seeds {start_seed} to {end_seed-1} ({n_seeds} seeds) ===\n")
    
    all_results = []
    metrics_by_seed = {
        'auc': [], 'accuracy': [], 'f1': [], 
        'precision': [], 'recall': [], 'oob_score': []
    }
    fold_wise_metrics = {1: [], 2: [], 3: [], 4: []}
    
    for seed in tqdm(range(start_seed, end_seed), desc="Testing RF seeds"):
        seed_results = train_rf_single_seed(handler, seed)
        
        avg_metrics = {}
        for metric in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
            avg_metrics[metric] = np.mean([f[metric] for f in seed_results])
            metrics_by_seed[metric].append(avg_metrics[metric])
        
        oob_scores = [f['oob_score'] for f in seed_results if f['oob_score'] is not None]
        if oob_scores:
            avg_metrics['oob_score'] = np.mean(oob_scores)
            metrics_by_seed['oob_score'].append(avg_metrics['oob_score'])
        
        all_results.append({
            'seed': seed,
            'avg_metrics': avg_metrics,
            'fold_results': seed_results
        })
        
        for fold_data in seed_results:
            fold_wise_metrics[fold_data['fold']].append(fold_data['auc'])
    
    return all_results, metrics_by_seed, fold_wise_metrics

def analyze_rf_stability(all_results, metrics_by_seed, fold_wise_metrics):
    """Analyze Random Forest stability across seeds"""
    
    print("\n" + "="*60)
    print("RANDOM FOREST SEED STABILITY ANALYSIS")
    print("="*60)
    
    # Overall stability for each metric
    print("\nOVERALL STABILITY BY METRIC:")
    print("-" * 40)
    
    stability_summary = {}
    for metric_name, values in metrics_by_seed.items():
        if values:  # Check if metric has values
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            cv = std_val / mean_val if mean_val != 0 else 0
            
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean:     {mean_val:.4f}")
            print(f"  Std Dev:  {std_val:.4f}")
            print(f"  Min:      {min_val:.4f}")
            print(f"  Max:      {max_val:.4f}")
            print(f"  Range:    {range_val:.4f}")
            print(f"  CV:       {cv:.4f}")
            
            stability_summary[metric_name] = {
                'mean': mean_val, 'std': std_val, 
                'min': min_val, 'max': max_val,
                'range': range_val, 'cv': cv
            }
    
    # Fold-wise stability (using AUC)
    print("\n" + "-" * 40)
    print("FOLD-WISE STABILITY (AUC):")
    print("-" * 40)
    
    for fold in range(1, 5):
        fold_aucs = fold_wise_metrics[fold]
        print(f"\nFold {fold}:")
        print(f"  Mean:     {np.mean(fold_aucs):.4f}")
        print(f"  Std Dev:  {np.std(fold_aucs):.4f}")
        print(f"  Range:    [{np.min(fold_aucs):.4f}, {np.max(fold_aucs):.4f}]")
    
    print("\n" + "-" * 40)
    print("STABILITY RANKING (by CV):")
    print("-" * 40)
    
    sorted_metrics = sorted(stability_summary.items(), key=lambda x: x[1]['cv'])
    for i, (metric, stats) in enumerate(sorted_metrics, 1):
        stability_label = "Most Stable" if i == 1 else ("Least Stable" if i == len(sorted_metrics) else "")
        print(f"{i}. {metric:12} CV={stats['cv']:.4f}  {stability_label}")
    
    # Check for concerning variations
    print("\n" + "-" * 40)
    print("STABILITY ASSESSMENT:")
    print("-" * 40)
    
    auc_cv = stability_summary['auc']['cv']
    if auc_cv < 0.01:
        print("✓ EXCELLENT: Model shows excellent stability (CV < 0.01)")
    elif auc_cv < 0.05:
        print("✓ GOOD: Model shows good stability (CV < 0.05)")
    elif auc_cv < 0.10:
        print("⚠ MODERATE: Model shows moderate stability (CV < 0.10)")
    else:
        print("✗ POOR: Model shows poor stability (CV >= 0.10)")
    
    return stability_summary

def visualize_rf_seed_stability(metrics_by_seed, fold_wise_metrics, all_results):
    """Create comprehensive visualization of RF seed stability"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. AUC distribution
    ax = axes[0, 0]
    ax.hist(metrics_by_seed['auc'], bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(metrics_by_seed['auc']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(metrics_by_seed["auc"]):.4f}')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    ax.set_title('AUC Distribution Across Seeds')
    ax.legend()
    
    # 2. AUC over seeds
    ax = axes[0, 1]
    seeds = list(range(len(metrics_by_seed['auc'])))
    ax.plot(seeds, metrics_by_seed['auc'], marker='o', markersize=3, alpha=0.7)
    mean_auc = np.mean(metrics_by_seed['auc'])
    std_auc = np.std(metrics_by_seed['auc'])
    ax.axhline(mean_auc, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(seeds, mean_auc - std_auc, mean_auc + std_auc, alpha=0.2, color='red')
    ax.set_xlabel('Seed Number')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Variation Across Seeds')
    ax.grid(True, alpha=0.3)
    
    # 3. Metrics comparison boxplot
    ax = axes[0, 2]
    metrics_data = [metrics_by_seed[m] for m in ['auc', 'accuracy', 'f1', 'precision', 'recall'] 
                    if m in metrics_by_seed and metrics_by_seed[m]]
    bp = ax.boxplot(metrics_data, labels=['AUC', 'Acc', 'F1', 'Prec', 'Rec'])
    ax.set_ylabel('Score')
    ax.set_title('Metrics Distribution')
    ax.grid(True, alpha=0.3)
    
    # 4. Fold-wise AUC boxplot
    ax = axes[1, 0]
    fold_data = [fold_wise_metrics[f] for f in range(1, 5)]
    bp = ax.boxplot(fold_data, labels=[f'Fold {i}' for i in range(1, 5)])
    ax.set_ylabel('AUC')
    ax.set_title('AUC Distribution by Fold')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 5. Correlation heatmap between folds
    ax = axes[1, 1]
    fold_matrix = np.array([fold_wise_metrics[f] for f in range(1, 5)])
    correlation = np.corrcoef(fold_matrix)
    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'F{i}' for i in range(1, 5)])
    ax.set_yticklabels([f'F{i}' for i in range(1, 5)])
    ax.set_title('Fold Performance Correlation')
    plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{correlation[i,j]:.2f}', ha='center', va='center')
    
    # 6. Stability vs Performance scatter
    ax = axes[1, 2]
    window_size = 10
    rolling_std = []
    rolling_mean = []
    for i in range(len(metrics_by_seed['auc']) - window_size + 1):
        window = metrics_by_seed['auc'][i:i+window_size]
        rolling_std.append(np.std(window))
        rolling_mean.append(np.mean(window))
    
    if rolling_std and rolling_mean:
        ax.scatter(rolling_std, rolling_mean, alpha=0.6)
        ax.set_xlabel('Local Std Dev (10-seed window)')
        ax.set_ylabel('Local Mean AUC')
        ax.set_title('Local Stability vs Performance')
        ax.grid(True, alpha=0.3)
    
    # 7. OOB Score distribution
    ax = axes[2, 0]
    if 'oob_score' in metrics_by_seed and metrics_by_seed['oob_score']:
        ax.hist(metrics_by_seed['oob_score'], bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(metrics_by_seed['oob_score']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(metrics_by_seed["oob_score"]):.4f}')
        ax.set_xlabel('OOB Score')
        ax.set_ylabel('Count')
        ax.set_title('Out-of-Bag Score Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No OOB data', ha='center', va='center')
        ax.set_title('Out-of-Bag Score Distribution')
    
    # 8. F1 Score variation
    ax = axes[2, 1]
    ax.plot(metrics_by_seed['f1'], marker='s', markersize=3, alpha=0.7, color='orange')
    ax.set_xlabel('Seed Number')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Across Seeds')
    ax.axhline(np.mean(metrics_by_seed['f1']), color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 9. Precision vs Recall scatter
    ax = axes[2, 2]
    ax.scatter(metrics_by_seed['precision'], metrics_by_seed['recall'], alpha=0.6)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision vs Recall Trade-off')
    ax.grid(True, alpha=0.3)
    
    # Add mean lines
    ax.axvline(np.mean(metrics_by_seed['precision']), color='red', linestyle='--', alpha=0.3)
    ax.axhline(np.mean(metrics_by_seed['recall']), color='red', linestyle='--', alpha=0.3)
    
    plt.suptitle('Random Forest Seed Stability Analysis (Seeds 0-50)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('rf_seed_stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to 'rf_seed_stability_analysis.png'")


def save_rf_results(all_results, metrics_by_seed):
    """Save detailed results to CSV"""
    
    # Create summary dataframe
    summary_data = []
    for result in all_results:
        seed_summary = {'seed': result['seed']}
        seed_summary.update(result['avg_metrics'])
        
        # Add fold-specific AUCs
        for fold_result in result['fold_results']:
            seed_summary[f'fold_{fold_result["fold"]}_auc'] = fold_result['auc']
        
        summary_data.append(seed_summary)
    
    df = pd.DataFrame(summary_data)
    df.to_csv('rf_seed_stability_results.csv', index=False)
    print("\nDetailed results saved to 'rf_seed_stability_results.csv'")
    
    # Print sample of results
    print("\nSample results (first 5 seeds):")
    print(df[['seed', 'auc', 'accuracy', 'f1']].head())
    
    return df

if __name__ == "__main__":

    handler = DataHandling()
    handler.load_data()
    
    all_results, metrics_by_seed, fold_wise_metrics = multi_seed_rf_analysis(
        handler, seed_range=(0, 51)
    )
    
    stability_summary = analyze_rf_stability(all_results, metrics_by_seed, fold_wise_metrics)
    
    visualize_rf_seed_stability(metrics_by_seed, fold_wise_metrics, all_results)
    
    results_df = save_rf_results(all_results, metrics_by_seed)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Random Forest tested with 51 different seeds (0-50)")
    print(f"Average AUC: {stability_summary['auc']['mean']:.4f} ± {stability_summary['auc']['std']:.4f}")
    print(f"AUC Range: [{stability_summary['auc']['min']:.4f}, {stability_summary['auc']['max']:.4f}]")
    print(f"Coefficient of Variation: {stability_summary['auc']['cv']:.4f}")