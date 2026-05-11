# CSCI 412 Project
# Professor Muath Obaidat
# By: Sayed Md Sajid Rahman (ID: 24183486)
#     Clara Muriel (ID: 23421111)
#     Miguel Diaz (ID: 24314665)

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, roc_curve

# legacy rule definitions

def legacy_rule_check_iiot(row):
    """Legacy rules looking for network telemetry spikes."""
    if row['SrcBytes'] > 1000: 
        return 1 
    if row['SrcLoad'] > 500000:
        return 1 
    return 0 

def legacy_rule_check_ehms(row):
    """Legacy rules looking for abnormal clinical vitals."""
    if row['Heart_rate'] > 120 or row['Heart_rate'] < 40:
        return 1 
    if row['SpO2'] < 90:
        return 1 
    return 0 

# robustness modifiers for data frames

def apply_df_noise(df, noise_factor=0.10):
    """Simulates 10% random fluctuation in sensor/network readings."""
    noisy_df = df.copy()
    numeric_cols = noisy_df.select_dtypes(include=np.number).columns
    noise = np.random.normal(1.0, noise_factor, size=noisy_df[numeric_cols].shape)
    noisy_df[numeric_cols] = noisy_df[numeric_cols] * noise
    return noisy_df

def apply_df_drift(df, drift_multiplier=1.20):
    """Simulates a 20% baseline increase due to a software update or new device."""
    drift_df = df.copy()
    numeric_cols = drift_df.select_dtypes(include=np.number).columns
    drift_df[numeric_cols] = drift_df[numeric_cols] * drift_multiplier
    return drift_df

# main pipeline

def run_baseline_pipeline(dataset_name, filepath, rule_func, target_col):
    print(f"\n{'='*50}")
    print(f"Executing Advanced Legacy Baseline for: {dataset_name}")
    print(f"{'='*50}")
    
    try:
        print(f"Loading {dataset_name} dataset...")
        df = pd.read_csv(filepath)
        y_true = df[target_col].values
        
        def evaluate_baseline(data_frame, title):
            preds = data_frame.apply(rule_func, axis=1).values
            roc_auc = roc_auc_score(y_true, preds)
            pr_auc = average_precision_score(y_true, preds)
            print(f"\n--- {title} ---")
            print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
            return preds

        # standard evaluation
        y_pred = evaluate_baseline(df, "Standard Baseline Evaluation")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # latency check
        print("\n--- Latency Check ---")
        single_row = df.iloc[0]
        start_time = time.perf_counter()
        _ = rule_func(single_row)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        print(f"Single Packet Inference Time: {latency_ms:.4f} ms")
        
        # robustness checks
        df_noisy = apply_df_noise(df)
        evaluate_baseline(df_noisy, "Robustness: Gaussian Noise (Sensor Variance)")
        
        df_drift = apply_df_drift(df)
        evaluate_baseline(df_drift, "Robustness: Concept Drift (Software Update)")
        
        # generate baseline visuals
        print("\nGenerating Baseline Visuals...")
        prefix = "EHMS" if "EHMS" in dataset_name else "IIoT"
        
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title(f'Baseline Confusion Matrix: {prefix}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{prefix}_baseline_confusion_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='red', lw=2, label=f'Baseline ROC AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Baseline ROC Curve: {prefix}')
        plt.legend(loc="lower right")
        plt.savefig(f'{prefix}_baseline_roc_curve.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visuals to '{prefix}_baseline_confusion_matrix.png' and '{prefix}_baseline_roc_curve.png'")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Skipping...")

# execute pipelines

if __name__ == "__main__":
    # run clinical vitals baseline
    run_baseline_pipeline(
        dataset_name="WUSTL-EHMS-2020",
        filepath='wustl-ehms-2020_with_attacks_categories.csv',
        rule_func=legacy_rule_check_ehms,
        target_col='Label'
    )
    
    # run network telemetry baseline
    run_baseline_pipeline(
        dataset_name="WUSTL-IIoT-2021",
        filepath='wustl_iiot_2021.csv',
        rule_func=legacy_rule_check_iiot,
        target_col='Target'
    )

print("\n--- end of legacy baseline IDS program ---")