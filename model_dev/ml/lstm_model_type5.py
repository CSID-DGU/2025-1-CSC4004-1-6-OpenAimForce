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


# 특징 추출 함수 정의
def extract_sequences_from_log(log_content_lines, player_ingame_id, max_seq_len=None):
    epsilon = 1e-9
    num_event_features = 15
    player_sequence = []
    prev_x, prev_y, prev_z = None, None, None
    prev_yaw, prev_pitch = None, None
    prev_t_microsec = None
    prev_omega_yaw, prev_omega_pitch = None, None
    prev_alpha_yaw, prev_alpha_pitch = None, None
    cumulative_shots_fired_by_player = 0
    cumulative_hits_given_by_player = 0

    for i, line in enumerate(log_content_lines):
        line = line.strip()
        current_event_features = np.zeros(num_event_features)
        event_occurred = False

        # 시간 파싱
        time_match = re.search(r't=(\d+)\s*µs', line)
        current_t_microsec = int(time_match.group(1)) if time_match else None

        # MOVE
        if line.startswith(f"[MOVE] {player_ingame_id}:"):
            event_occurred = True
            pos_match = re.search(r'x=([\d\.-]+),\s*y=([\d\.-]+),\s*z=([\d\.-]+)', line)
            ang_match = re.search(r'yaw=(\d+),\s*pitch=(-?\d+)', line)

            if pos_match and ang_match and current_t_microsec is not None:
                current_x, current_y, current_z = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                current_yaw, current_pitch = int(ang_match.group(1)), int(ang_match.group(2))

                # delta_yaw, delta_pitch 계산
                if prev_yaw is not None and prev_pitch is not None:
                    # Yaw 변화량
                    delta_yaw_event = current_yaw - prev_yaw
                    if delta_yaw_event > 180: delta_yaw_event -= 360
                    elif delta_yaw_event < -180: delta_yaw_event += 360
                    current_event_features[0] = delta_yaw_event

                    # Pitch 변화량
                    delta_pitch_event = current_pitch - prev_pitch
                    current_event_features[1] = delta_pitch_event

                # dt
                if prev_t_microsec is not None and current_t_microsec > prev_t_microsec:
                    dt = (current_t_microsec - prev_t_microsec) / 1_000_000.0
                    if dt < 1e-9: dt = 1e-9

                    # 1. 3D 이동 속도
                    if prev_x is not None:
                        dx, dy, dz = current_x - prev_x, current_y - prev_y, current_z - prev_z
                        current_event_features[6] = np.sqrt(dx**2 + dy**2 + dz**2) / dt

                    # 2. Yaw/Pitch 각속도
                    omega_y = current_event_features[0] / dt
                    omega_p = current_event_features[1] / dt
                    current_event_features[7] = omega_y
                    current_event_features[8] = omega_p

                    # 3. Yaw/Pitch 각가속도
                    alpha_y = None
                    alpha_p = None
                    if prev_omega_yaw is not None:
                        alpha_y = (omega_y - prev_omega_yaw) / dt
                        alpha_p = (omega_p - prev_omega_pitch) / dt
                        current_event_features[9] = alpha_y
                        current_event_features[10] = alpha_p

                    #4. Yaw/Pitch 각저크
                    if prev_alpha_yaw is not None:
                             current_event_features[11] = (alpha_y - prev_alpha_yaw) / dt
                             current_event_features[12] = (alpha_p - prev_alpha_pitch) / dt

                    #5. Pitch 대 Yaw 각속도 비율
                    current_event_features[13] = abs(omega_p) / (abs(omega_y) + epsilon)

                    prev_omega_yaw, prev_omega_pitch = omega_y, omega_p
                    prev_alpha_yaw, prev_alpha_pitch = alpha_y, alpha_p

                prev_x, prev_y, prev_z = current_x, current_y, current_z
                prev_yaw, prev_pitch = current_yaw, current_pitch

        if current_t_microsec is not None:
              prev_t_microsec = current_t_microsec


        # SHOOT
        elif line.startswith(f"[SHOOT] {player_ingame_id}:"):
            # current_event_features[2] = 1
            cumulative_shots_fired_by_player += 1
            event_occurred = True
            if current_t_microsec is not None: prev_t_microsec = current_t_microsec

        # HIT
        elif line.startswith(f"[HIT] {player_ingame_id}:"):
            # current_event_features[3] = 1
            cumulative_hits_given_by_player += 1
            event_occurred = True
            if current_t_microsec is not None: prev_t_microsec = current_t_microsec

        # DEAD
        elif line.startswith(f"[DEAD] {player_ingame_id}:"):
            # current_event_features[4] = 1
            event_occurred = True
            if current_t_microsec is not None: prev_t_microsec = current_t_microsec

        # KILL
        elif line.startswith("[DEAD]"):
            if i + 1 < len(log_content_lines):
                next_line = log_content_lines[i+1].strip()
                kill_match = re.match(r'\[.*?\]\s+' + re.escape(player_ingame_id) + r'\s+\w+\s+\S+', next_line)
                if kill_match:
                    # current_event_features[5] = 1
                    event_occurred = True

        if cumulative_shots_fired_by_player > 0:
            current_event_features[14] = cumulative_hits_given_by_player / cumulative_shots_fired_by_player
        else:
            current_event_features[14] = 0.0

        if event_occurred:
            player_sequence.append(current_event_features)

    if max_seq_len and len(player_sequence) > max_seq_len:
        player_sequence = player_sequence[:max_seq_len]

    return player_sequence


# 1. DB 로드
print("Step 1: Loading original data...")
df_original = pd.read_csv("game_db.csv")

HACK_TYPES_TO_MODEL = [5] #[2, 3, 4, 5]
results_summary = {}

# 각 핵 타입에 대해 모델 학습 및 평가
for target_hack_type in HACK_TYPES_TO_MODEL:
    print(f"\n======================================================================")
    print(f" Processing: Normal (aimhack_type 1) vs. Hack Type {target_hack_type} ")
    print(f"======================================================================")

    # 1.1. 데이터 필터링 및 라벨링
    print(f"\n[Task {target_hack_type}] Step 1.1: Filtering data and preparing labels...")
    if target_hack_type != 5:
        df_filtered = df_original[
            (df_original['aimhack'] == 1) | (df_original['aimhack'] == target_hack_type)
        ].copy()
    else:
        df_filtered = df_original.copy()
    df_filtered['label'] = df_filtered['aimhack'].apply(lambda x: 0 if x == 1 else 1)

    print(f"Filtered data shape for task {target_hack_type}: {df_filtered.shape}")
    print(f"Label distribution for task {target_hack_type}:\n{df_filtered['label'].value_counts()}")

    # 1.2. 시퀀스 추출
    print(f"\n[Task {target_hack_type}] Step 1.2: Extracting sequences...")
    all_sequences_task = []
    all_labels_task = []
    log_file_base_path = "."

    for index, row in df_filtered.iterrows():
        player_id = row['ingame_id']
        label = row['label']
        logfile_name_from_csv = row['logfile_name']

        if not isinstance(logfile_name_from_csv, str) or not logfile_name_from_csv.strip():
            continue
        actual_log_filename_on_disk = "game_" + logfile_name_from_csv
        log_file_full_path = os.path.join(log_file_base_path, actual_log_filename_on_disk)

        log_content_lines = None
        try:
            with open(log_file_full_path, 'r', encoding='utf-8') as f:
                log_content_lines = f.readlines()
        except FileNotFoundError:
            print(f"Log file not found: {log_file_full_path}")
            continue
        except Exception as e:
            print(f"Error reading log file {log_file_full_path}: {e}")
            continue

        if log_content_lines:
            try:
                sequence = extract_sequences_from_log(log_content_lines, player_id, max_seq_len=500)
                if sequence:
                    all_sequences_task.append(sequence)
                    all_labels_task.append(label)
            except Exception as e:
                print(f"Error extracting sequence for {player_id} in {actual_log_filename_on_disk}: {e}")
                pass


    print(f"Extracted {len(all_sequences_task)} sequences for task {target_hack_type}.")

    # 1.3. 데이터 패딩 및 분할
    print(f"\n[Task {target_hack_type}] Step 1.3: Padding and splitting data...")
    X_padded_task = pad_sequences(all_sequences_task, padding='post', dtype='float32', value=PADDING_VALUE)
    y_array_task = np.array(all_labels_task)

    # stratify 옵션을 위한 최소 샘플 수 확인
    min_samples_for_stratify = 2 * len(np.unique(y_array_task))
    stratify_option_task = y_array_task if len(y_array_task) >= min_samples_for_stratify and X_padded_task.shape[0] >= min_samples_for_stratify else None

    X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(
        X_padded_task, y_array_task, test_size=0.2, random_state=42, stratify=stratify_option_task
    )
    print(f"X_train shape: {X_train_task.shape}, y_train shape: {y_train_task.shape}")

    # 1.4. 특징 스케일링 StandardScaler
    print(f"\n[Task {target_hack_type}] Step 1.4: Applying Feature Scaling...")
    num_event_features = 15
    # delta_yaw, delta_pitch, speed_3d, omegas, alphas, jerks, ratio, cumulative_accuracy
    feature_indices_to_scale_task = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14] 
    scalers_task = {} 
    if X_train_task.shape[0] > 0 and X_train_task.shape[2] == num_event_features:
        for f_idx in feature_indices_to_scale_task:
            if f_idx >= X_train_task.shape[2]:
                print(f"Warning: Feature index {f_idx} is out of bounds for scaling. Skipping.")
                continue

            scaler = StandardScaler()
            feature_data_train = X_train_task[:, :, f_idx]
            
            actual_data_mask_train = (feature_data_train != PADDING_VALUE)
            values_to_fit_scaler = feature_data_train[actual_data_mask_train].reshape(-1, 1)
            
            if values_to_fit_scaler.size > 0: 
                scaler.fit(values_to_fit_scaler)
                scalers_task[f_idx] = scaler
                scaled_values_train = scaler.transform(values_to_fit_scaler)
                X_train_task[:, :, f_idx][actual_data_mask_train] = scaled_values_train.flatten() 

                if X_test_task.shape[0] > 0:
                    feature_data_test = X_test_task[:, :, f_idx]
                    actual_data_mask_test = (feature_data_test != PADDING_VALUE)
                    values_to_transform_test = feature_data_test[actual_data_mask_test].reshape(-1, 1)

                    if values_to_transform_test.size > 0:
                        scaled_values_test = scaler.transform(values_to_transform_test)
                        X_test_task[:, :, f_idx][actual_data_mask_test] = scaled_values_test.flatten()
                print(f"  Feature at index {f_idx} scaled for task {target_hack_type}.")
            else:
                print(f"  No actual data (all padding or empty) for feature {f_idx} in X_train_task for task {target_hack_type}. Scaling skipped.")
                scalers_task[f_idx] = None
    else:
        print(f"X_train_task is empty or feature dimension mismatch for task {target_hack_type}. Skipping feature scaling.")


    # 1.5. 클래스 가중치 계산
    print(f"\n[Task {target_hack_type}] Step 1.5: Calculating class weights...")
    if y_train_task.size > 0 and len(np.unique(y_train_task)) > 1:
        task_weights = class_weight.compute_class_weight('balanced',
                                                         classes=np.unique(y_train_task),
                                                         y=y_train_task)
        task_class_weights_dict = dict(enumerate(task_weights))
        print(f"Calculated class weights for task {target_hack_type}: {task_class_weights_dict}")
    else:
        task_class_weights_dict = None


    # 2. 모델 정의 및 컴파일
    print(f"\n[Task {target_hack_type}] Step 2: Defining and compiling model...")
    num_features_per_step_task = X_train_task.shape[2]

    l2_lambda = 0.00001

    model = Sequential([
        tf.keras.Input(shape=(None, num_features_per_step_task), name="input_sequence"),
        Masking(mask_value=PADDING_VALUE),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))),
        Dropout(0.5),
        Bidirectional(LSTM(96, return_sequences=False, kernel_regularizer=regularizers.l2(l2_lambda))),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-7
    )
    callbacks_list = [early_stopping, reduce_lr]

    print("\nStarting LSTM model training...")
    history_task = model.fit(X_train_task, y_train_task, epochs=300, batch_size=32, validation_split=0.2, callbacks=callbacks_list, verbose=1)
    print(f"\n--- Training finished for task {target_hack_type} ---")

    # 3. 결과 저장 및 시각화
    print(f"\n[Task {target_hack_type}] Step 3: Storing and plotting results...")
    eval_results = model.evaluate(X_test_task, y_test_task, verbose=0)
    y_pred_probs_task = model.predict(X_test_task, verbose=0)
    y_pred_classes_task = (y_pred_probs_task > 0.5).astype(int)

    cm = confusion_matrix(y_test_task, y_pred_classes_task)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    results_summary[target_hack_type] = {
        'history': history_task.history,
        'best_val_loss': np.min(history_task.history['val_loss']) if 'val_loss' in history_task.history else float('inf'),
        'best_val_accuracy': np.max(history_task.history['val_accuracy']) if 'val_accuracy' in history_task.history else float('-inf'),
        'test_loss': eval_results[0],
        'test_accuracy': eval_results[1],
        'test_precision': eval_results[2],
        'test_recall': eval_results[3],
        'test_f1_score': f1_score(y_test_task, y_pred_classes_task, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

    plt.figure(figsize=(14, 6))
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_task.history['loss'], label=f'Train Loss (Type {target_hack_type})')
    plt.plot(history_task.history['val_loss'], label=f'Val Loss (Type {target_hack_type})')
    plt.title(f'Loss for Normal vs. Hack Type {target_hack_type}')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history_task.history['accuracy'], label=f'Train Acc (Type {target_hack_type})')
    plt.plot(history_task.history['val_accuracy'], label=f'Val Acc (Type {target_hack_type})')
    plt.title(f'Accuracy for Normal vs. Hack Type {target_hack_type}')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.suptitle(f"Results for Normal (1) vs. Hack Type {target_hack_type}", fontsize=16, y=1.02)
    # plt.savefig(f"results_hack_type_{target_hack_type}.png")
    plt.show()

    model_save_filename = f"aimhack_model_type_{target_hack_type}.keras"
    print(f"\n[Task {target_hack_type}] Step 6: Saving the trained model to {model_save_filename} ...")
    try:
        model.save(model_save_filename)
        print(f"Model for hack type {target_hack_type} saved successfully as {model_save_filename}")
    except Exception as e:
        print(f"Error saving model for hack type {target_hack_type}: {e}")

print("\n\n======================================================================")
print("                      Model Evaluation Summary                      ")
print("======================================================================")
table_data = []
for hack_type, metrics in results_summary.items():
    if metrics.get('status') == 'skipped':
        table_data.append({
            'Hack Type vs Normal (1)': f"Type {hack_type}",
            'Status': 'Skipped',
            'Reason': metrics.get('reason', '')
        })
    else:
        table_data.append({
            'Hack Type vs Normal': f"Type {hack_type}",
            'Accuracy': f"{metrics.get('test_accuracy', 0):.4f}",
            'Precision': f"{metrics.get('test_precision', 0):.4f}",
            'Recall': f"{metrics.get('test_recall', 0):.4f}",
            'F1-Score': f"{metrics.get('test_f1_score', 0):.4f}",
            'TN': metrics.get('TN', 0),
            'FP': metrics.get('FP', 0),
            'FN': metrics.get('FN', 0),
            'TP': metrics.get('TP', 0),
            'Best_Val_Loss': f"{metrics.get('best_val_loss', [float('inf')]):.4f}",
            'Best_Val_Accuracy': f"{metrics.get('best_val_accuracy', [float('inf')]):.4f}"
        })

summary_df = pd.DataFrame(table_data)
if not summary_df.empty:
    desired_columns = ['Hack Type vs Normal', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'TN', 'FP', 'FN', 'TP', 'Best_Val_Loss', 'Best_Val_Accuracy']
    ordered_columns = [col for col in desired_columns if col in summary_df.columns]
    summary_df = summary_df[ordered_columns]

print(summary_df.to_string())
print("======================================================================")
