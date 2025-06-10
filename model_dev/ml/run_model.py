import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import re

MODEL_FILENAME = "result/final_aimhack_model.keras" 

def extract_sequences_from_log(log_content_lines, player_id, max_seq_len=None):
    player_id = "player_pid_" + str(player_id)
    epsilon = 1e-9
    num_event_features = 8
    player_sequence = []
    
    prev_state = {'x': None, 'yaw': None, 'pitch': None, 
                  't_microsec': None, 'omega_yaw': None, 'omega_pitch': None, 
                  'alpha_yaw': None, 'alpha_pitch': None}
    
    cumulative_shots = 0
    cumulative_hits = 0

    for i, line in enumerate(log_content_lines):
        line = line.strip()
        features = np.zeros(num_event_features)
        event_occurred = False
        
        time_match = re.search(r't=(\d+)\s*µs', line)
        current_t = int(time_match.group(1)) if time_match else None

        if line.startswith(f"[MOVE] {player_id}:"):
            event_occurred = True
            pos_match = re.search(r'x=([\d\.-]+),\s*y=([\d\.-]+),\s*z=([\d\.-]+)', line)
            ang_match = re.search(r'yaw=(\d+),\s*pitch=(-?\d+)', line)

            if pos_match and ang_match and current_t is not None:
                state = {'x': float(pos_match.group(1)), 'y': float(pos_match.group(2)), 'z': float(pos_match.group(3)),
                         'yaw': int(ang_match.group(1)), 'pitch': int(ang_match.group(2)), 't_microsec': current_t}
                
                if prev_state['t_microsec'] is not None and state['t_microsec'] > prev_state['t_microsec']:
                    dt = (state['t_microsec'] - prev_state['t_microsec']) / 1_000_000.0
                    if dt < epsilon: dt = epsilon
                    
                    delta_yaw = state['yaw'] - prev_state['yaw']
                    if delta_yaw > 180: delta_yaw -= 360
                    elif delta_yaw < -180: delta_yaw += 360
                    delta_pitch = state['pitch'] - prev_state['pitch']
                    
                    omega_y, omega_p = delta_yaw / dt, delta_pitch / dt
                    features[0], features[1] = omega_y, omega_p
                    
                    alpha_y, alpha_p = 0.0, 0.0
                    if prev_state['omega_yaw'] is not None:
                        alpha_y, alpha_p = (omega_y - prev_state['omega_yaw']) / dt, (omega_p - prev_state['omega_pitch']) / dt
                        features[2], features[3] = alpha_y, alpha_p
                        
                        if prev_state['alpha_yaw'] is not None:
                            features[4] = (alpha_y - prev_state['alpha_yaw']) / dt
                    
                    features[5] = abs(omega_p) / (abs(omega_y) + epsilon)
                    prev_state['omega_yaw'], prev_state['omega_pitch'] = omega_y, omega_p
                    prev_state['alpha_yaw'], prev_state['alpha_pitch'] = alpha_y, alpha_p
                
                prev_state.update({'x': state['x'], 'y': state['y'], 'z': state['z'],
                                   'yaw': state['yaw'], 'pitch': state['pitch']})

        elif line.startswith(f"[SHOOT] {player_id}:"):
            cumulative_shots += 1; event_occurred = True
        elif line.startswith(f"[HIT] {player_id}:"):
            features[7] = 1; cumulative_hits += 1; event_occurred = True
        elif line.startswith(f"[DEAD] {player_id}:"):
            event_occurred = True 
        elif line.startswith("[DEAD]"):
            if i + 1 < len(log_content_lines) and re.match(r'\[.*?\]\s+' + re.escape(player_id) + r'\s+\w+\s+\S+', log_content_lines[i+1].strip()):
                event_occurred = True
        
        if event_occurred:
            features[6] = cumulative_hits / cumulative_shots if cumulative_shots > 0 else 0.0
            player_sequence.append(features)
            if current_t is not None: prev_state['t_microsec'] = current_t

    return player_sequence[:max_seq_len] if max_seq_len else player_sequence

print("Step 1: Loading and Preparing Data...")
df_original = pd.read_csv("../dataset/game_db_6.csv")

df_filtered = df_original.copy()
df_filtered['label'] = df_original['aimhack'].apply(lambda x: 0 if x == 1 else 1)
print(f"Data shape: {df_filtered.shape}, Label distribution:\n{df_filtered['label'].value_counts()}")

print("\nStep 2: Extracting Sequences...")
all_sequences = []
all_labels = []
log_file_base_path = "../dataset/id_stripped/"
for index, row in df_filtered.iterrows():
    player_id = row['pid']
    label = row['label']
    logfile_name = row['logfile_name']

    if not isinstance(logfile_name, str) or not logfile_name.strip(): continue;
    
    log_file_full_path = os.path.join(log_file_base_path, "game_" + logfile_name)
    try:
        with open(log_file_full_path, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"Error reading {log_file_full_path}: {e}")
        continue
    sequence = extract_sequences_from_log(log_lines, player_id, max_seq_len=500)
    if sequence:
        all_sequences.append(sequence)
        all_labels.append(label)
print(f"Extracted {len(all_sequences)} sequences.")

print("\nStep 3: Padding, Splitting, and Scaling Data...")
X_padded = pad_sequences(all_sequences, padding='post', dtype='float32', value=PADDING_VALUE)
y_array = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_array, test_size=0.2, random_state=3, 
    stratify=y_array if len(np.unique(y_array)) > 1 and np.all(np.bincount(y_array) >= 2) else None
)
num_features = X_train.shape[2]
feature_indices_to_scale = [i for i in range(num_features) if i not in [7, 9]]
scalers = {}
for f_idx in feature_indices_to_scale:
    scaler = StandardScaler()
    actual_data_mask = (X_train[:, :, f_idx] != PADDING_VALUE)
    values_to_fit = X_train[:, :, f_idx][actual_data_mask].reshape(-1, 1)
    if values_to_fit.size > 0:
        scaler.fit(values_to_fit)
        scalers[f_idx] = scaler
        X_train[:, :, f_idx][actual_data_mask] = scaler.transform(values_to_fit).flatten()
        
        actual_data_mask_test = (X_test[:, :, f_idx] != PADDING_VALUE)
        values_to_transform_test = X_test[:, :, f_idx][actual_data_mask_test].reshape(-1, 1)
        if values_to_transform_test.size > 0:
            X_test[:, :, f_idx][actual_data_mask_test] = scaler.transform(values_to_transform_test).flatten()
print("Feature scaling applied.")

if 'X_test' not in locals() or 'y_test' not in locals():
    exit() 

print(f"Step 4: Loading the saved model from '{MODEL_FILENAME}'...")
try:
    loaded_model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully.")
    loaded_model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

eval_results = loaded_model.evaluate(X_test, y_test, verbose=1)


print("\nStep 5: Generating predictions and calculating detailed metrics...")
y_pred_probs = loaded_model.predict(X_test, verbose=0)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, zero_division=0)
recall = recall_score(y_test, y_pred_classes, zero_division=0)
f1 = f1_score(y_test, y_pred_classes, zero_division=0)
cm = confusion_matrix(y_test, y_pred_classes)

try:
    tn, fp, fn, tp = cm.ravel()
except ValueError:
    tn, fp, fn, tp = 0, 0, 0, 0 
    if len(np.unique(y_test)) == 1: 
         if np.unique(y_test)[0] == 0 and len(np.unique(y_pred_classes)) == 1 and np.unique(y_pred_classes)[0] == 0:
             tn = len(y_test)
         elif np.unique(y_test)[0] == 1 and len(np.unique(y_pred_classes)) == 1 and np.unique(y_pred_classes)[0] == 1:
             tp = len(y_test)

print("\n======================================================================")
print("              Final Model Evaluation on Test Data                 ")
print("======================================================================")
print(f"Model File: {MODEL_FILENAME}")
print(f"Test Data Shape: {X_test.shape}")
print(f"Test Label Distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

print("\n--- Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n--- Confusion Matrix ---")
print("                Predicted")
print("             Normal(0)  Hack(1)")
print(f"Actual Normal(0) |  {tn:<7d}  |  {fp:<7d} |")
print(f"Actual Hack(1)   |  {fn:<7d}  |  {tp:<7d} |")
print("---------------------------------")
print(f"TN (True Negative):  {tn} - 실제 '정상'을 '정상'으로 올바르게 예측")
print(f"FP (False Positive): {fp} - 실제 '정상'을 '핵'으로 잘못 예측")
print(f"FN (False Negative): {fn} - 실제 '핵'을 '정상'으로 잘못 예측 (놓침)")
print(f"TP (True Positive):  {tp} - 실제 '핵'을 '핵'으로 올바르게 예측")
print("======================================================================")

table_data = [{
    'Accuracy': f"{accuracy:.4f}",
    'Precision': f"{precision:.4f}",
    'Recall': f"{recall:.4f}",
    'F1-Score': f"{f1:.4f}",
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
}]
summary_df = pd.DataFrame(table_data)
print("\n--- Summary Table ---")
print(summary_df.to_string(index=False))
