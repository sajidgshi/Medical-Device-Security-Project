# CSCI 412 Project
# Professor Muath Obaidat
# By: Sayed Md Sajid Rahman (ID: 24183486)
#     Clara Muriel (ID: 23421111)
#     Miguel Diaz (ID: 24314665)

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, roc_curve

# robustness attack functions

def apply_gaussian_noise(data, noise_factor=0.1):
    """Simulates messy sensor data or unstable network connections."""
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    return np.clip(data + noise, 0., 1.)

def apply_concept_drift(data, drift_factor=0.2):
    """Simulates a device software update that shifts normal behavior baselines."""
    return np.clip(data + drift_factor, 0., 1.)

def generate_fgsm_adversarial_noise(model, x_data, epsilon=0.05):
    """
    White-box adversarial attack (Fast Gradient Sign Method).
    Calculates the gradient of the reconstruction loss and adds noise in that direction
    to force the autoencoder to misclassify normal data as an anomaly.
    """
    x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        reconstruction = model(x_tensor)
        loss = tf.reduce_mean(tf.square(x_tensor - reconstruction))
    
    gradient = tape.gradient(loss, x_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_x = x_tensor + epsilon * signed_grad
    return np.clip(adversarial_x.numpy(), 0., 1.)

# main pipeline

def run_security_pipeline(dataset_name, filepath, features, target_col):
    print(f"\n{'='*50}")
    print(f"Executing Pulse Check Pipeline for: {dataset_name}")
    print(f"{'='*50}")

    # load data
    df = pd.read_csv(filepath)
    df_normal = df[df[target_col] == 0].copy()
    df_attack = df[df[target_col] == 1].copy()
    
    # scale data
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(df_normal[features])
    X_attack_scaled = scaler.transform(df_attack[features])

    def create_sequences(data, time_steps=5):
        Xs = []
        for i in range(len(data) - time_steps):
            Xs.append(data[i:(i + time_steps)])
        return np.array(Xs)

    TIME_STEPS = 5 
    X_train = create_sequences(X_normal_scaled, TIME_STEPS)

    # create a balanced test set
    test_size = min(1000, len(X_attack_scaled) - TIME_STEPS, len(X_normal_scaled) - 2000)
    X_test_data = np.vstack([X_normal_scaled[-test_size:], X_attack_scaled[:test_size]])
    y_test = np.array([0]*test_size + [1]*test_size)
    X_test = create_sequences(X_test_data, TIME_STEPS)
    y_test = y_test[TIME_STEPS:]

    # build & train LSTM
    print("Building and Training Sentinel LSTM...")
    model = Sequential([
        LSTM(16, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        RepeatVector(X_train.shape[1]),
        LSTM(16, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X_train.shape[2]))
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # we kept the epochs low for speed 
    model.fit(X_train, X_train, epochs=2, batch_size=128, validation_split=0.1, verbose=0)

    # calculating threshold
    train_reconstructions = model.predict(X_train[:2000], verbose=0)
    train_mse = np.mean(np.mean(np.power(X_train[:2000] - train_reconstructions, 2), axis=2), axis=1)
    THRESHOLD = np.percentile(train_mse, 95)

    def evaluate_model(X, y_true, title):
        reconstructions = model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=2)
        anomaly_scores = np.mean(mse, axis=1)
        preds = (anomaly_scores > THRESHOLD).astype(int)
        
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        pr_auc = average_precision_score(y_true, anomaly_scores)
        print(f"\n--- {title} ---")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return preds, anomaly_scores

    # standard evaluation
    preds, scores = evaluate_model(X_test, y_test, "Proposed Model: Standard Evaluation")
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # chart generation
    print("\nGenerating visual plots...")
    prefix = "EHMS" if "EHMS" in dataset_name else "IIoT"
    
    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix: {prefix} Dataset')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{prefix}_confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {prefix} Dataset')
    plt.legend(loc="lower right")
    plt.savefig(f'{prefix}_roc_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {prefix}_confusion_matrix.png and {prefix}_roc_curve.png")

    # latency check
    print("\n--- Latency Check ---")
    single_sample = X_test[0:1] 
    _ = model.predict(single_sample, verbose=0) 
    
    start_time = time.perf_counter()
    _ = model.predict(single_sample, verbose=0)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    print(f"Single Packet Inference Time: {latency_ms:.2f} ms")
    if latency_ms < 50:
        print("STATUS: PASSED (<50ms target met)")
    else:
        print("STATUS: FAILED (Exceeded 50ms target)")

    # noise robustness check
    X_test_noisy = apply_gaussian_noise(X_test)
    evaluate_model(X_test_noisy, y_test, "Robustness: Gaussian Noise (Sensor Variance)")

    # drift robustness check
    X_test_drift = apply_concept_drift(X_test)
    evaluate_model(X_test_drift, y_test, "Robustness: Concept Drift (Software Update)")

    # adversarial robustness check
    X_test_normal_only = X_test[y_test == 0]
    y_test_normal_only = y_test[y_test == 0]
    X_test_fgsm = generate_fgsm_adversarial_noise(model, X_test_normal_only)
    evaluate_model(X_test_fgsm, y_test_normal_only, "Robustness: FGSM Adversarial Attack")

# execute pipelines

if __name__ == "__main__":
    # clinical vitals dataset
    ehms_features = ['Temp', 'SpO2', 'Pulse_Rate', 'Heart_rate', 'Resp_Rate']
    try:
        run_security_pipeline(
            dataset_name="WUSTL-EHMS-2020 (Clinical Vitals)",
            filepath='wustl-ehms-2020_with_attacks_categories.csv',
            features=ehms_features,
            target_col='Label' 
        )
    except FileNotFoundError:
        print("Could not find EHMS dataset. Skipping...")

    # network telemetry dataset
    iiot_features = ['SrcBytes', 'DstBytes', 'SrcPkts', 'DstPkts', 'SIntPkt', 'DIntPkt', 'SrcLoad', 'DstLoad']
    try:
        run_security_pipeline(
            dataset_name="WUSTL-IIoT-2021 (Network Telemetry)",
            filepath='wustl_iiot_2021.csv',
            features=iiot_features,
            target_col='Target' 
        )
    except FileNotFoundError:
        print("Could not find IIoT dataset. Skipping...")