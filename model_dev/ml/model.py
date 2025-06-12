import pandas as pd
import numpy as np
import tensorflow as tf
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

PADDING_VALUE = -999.0

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

    if not isinstance(logfile_name, str) or not logfile_name.strip(): continue
    
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

class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_dict))
print(f"Calculated class weights: {class_weights_dict}")

print("\nStep 4: Defining and Compiling Model...")
l2_lambda = 0.003
model = Sequential([
    Input(shape=(None, num_features), name="input_sequence"),
    Masking(mask_value=PADDING_VALUE),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(l2_lambda))),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
opt = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
model.summary()

print("\nStep 5: Training Model...")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)
history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                    validation_split=0.2, 
                    callbacks=[reduce_lr],
                    class_weight=class_weights_dict,
                    verbose=1)
print("--- Training Finished ---")

print("\nStep 6: Evaluating, Plotting, and Saving...")
eval_results = model.evaluate(X_test, y_test, verbose=0)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_classes)
tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
results = {
    'Accuracy': eval_results[1], 'Precision': eval_results[2], 'Recall': eval_results[3],
    'F1-Score': f1_score(y_test, y_pred_classes, zero_division=0),
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    'Best Val Loss': min(history.history.get('val_loss', [np.inf])),
    'Best Val Accuracy': max(history.history.get('val_accuracy', [0]))
}
summary_df = pd.DataFrame([results])
print(summary_df.to_string())

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.suptitle('Training and Validation Metrics', fontsize=16)
plt.show()

model_save_filename = "result/final_aimhack_model.keras"
model.save(model_save_filename)
print(f"\nModel saved successfully as {model_save_filename}")
