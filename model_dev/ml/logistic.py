from sklearn.linear_model import LogisticRegression

def create_summary_features(sequence, feature_names):
    summary = {}
    if not sequence or len(sequence) == 0:
        return summary

    seq_array = np.array(sequence)

    for i, name in enumerate(feature_names):
        feature_column = seq_array[:, i]
        summary[f'{name}_mean'] = np.mean(feature_column)
        summary[f'{name}_std'] = np.std(feature_column)
        summary[f'{name}_max'] = np.max(feature_column)
        summary[f'{name}_median'] = np.median(feature_column)
        
    for i, name in enumerate(feature_names):
        if name.startswith('is_'):
            summary[f'{name}_sum'] = np.sum(seq_array[:, i])
            
    return summary


def extract_sequences_from_log(log_content_lines, player_ingame_id, max_seq_len=None):
    epsilon = 1e-9
    num_event_features = 10
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

        if line.startswith(f"[MOVE] {player_ingame_id}:"):
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
                            features[5] = (alpha_p - prev_state['alpha_pitch']) / dt
                    
                    features[6] = abs(omega_p) / (abs(omega_y) + epsilon)
                    prev_state['omega_yaw'], prev_state['omega_pitch'] = omega_y, omega_p
                    prev_state['alpha_yaw'], prev_state['alpha_pitch'] = alpha_y, alpha_p
                
                prev_state.update({'x': state['x'], 'y': state['y'], 'z': state['z'],
                                   'yaw': state['yaw'], 'pitch': state['pitch']})

        elif line.startswith(f"[SHOOT] {player_ingame_id}:"):
            cumulative_shots += 1; event_occurred = True
        elif line.startswith(f"[HIT] {player_ingame_id}:"):
            features[9] = 1; cumulative_hits += 1; event_occurred = True
        elif line.startswith(f"[DEAD] {player_ingame_id}:"):
            event_occurred = True 
        elif line.startswith("[DEAD]"):
            if i + 1 < len(log_content_lines) and re.match(r'\[.*?\]\s+' + re.escape(player_ingame_id) + r'\s+\w+\s+\S+', log_content_lines[i+1].strip()):
                features[7] = 1; event_occurred = True
        
        if event_occurred:
            features[8] = cumulative_hits / cumulative_shots if cumulative_shots > 0 else 0.0
            player_sequence.append(features)
            if current_t is not None: prev_state['t_microsec'] = current_t

    return player_sequence[:max_seq_len] if max_seq_len else player_sequence

print("Step 1: Loading and Preparing Data...")
df_original = pd.read_csv("game_db_6.csv")
df_filtered = df_original.copy()
df_filtered['label'] = df_original['aimhack'].apply(lambda x: 0 if x == 1 else 1)
print(f"Data shape: {df_filtered.shape}, Label distribution:\n{df_filtered['label'].value_counts()}")

print("\nStep 2: Extracting Sequences and Creating Summary Features for Logistic Regression...")
all_summary_features = []
all_labels_logistic = []
log_file_base_path = "."

feature_names_in_sequence = [
    'omega_yaw', 'omega_pitch', 'alpha_yaw', 'alpha_pitch', 'jerk_yaw', 'jerk_pitch',
    'pitch_yaw_omega_ratio', 'is_player_kill', 'cumulative_accuracy', 'is_hit_given'
]

for index, row in df_filtered.iterrows():
    player_id = row['ingame_id']
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
        summary_dict = create_summary_features(sequence, feature_names_in_sequence)
        all_summary_features.append(summary_dict)
        all_labels_logistic.append(label)

print(f"Extracted {len(all_summary_features)} summary feature sets.")

print("\nStep 3: Creating DataFrame and Splitting Data...")
X_logistic_df = pd.DataFrame(all_summary_features)
y_logistic_array = np.array(all_labels_logistic)

X_logistic_df = X_logistic_df.fillna(0)

print(f"Logistic Regression Feature Matrix Shape: {X_logistic_df.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_logistic_df, y_logistic_array, test_size=0.2, random_state=42, 
    stratify=y_logistic_array if len(np.unique(y_logistic_array)) > 1 and np.all(np.bincount(y_logistic_array) >= 2) else None
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling applied.")

print("\nStep 4: Training Logistic Regression Model...")
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
print("--- Training Finished ---")

print("\nStep 5: Evaluating Model...")
y_pred = lr_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

print("\n======================================================================")
print("            Logistic Regression Evaluation Summary              ")
print("======================================================================")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print(f"Confusion Matrix:\n  TN: {tn}, FP: {fp}\n  FN: {fn}, TP: {tp}")
print("======================================================================")

print("\n--- Feature Importance (Coefficients) ---")
feature_importance = pd.DataFrame({
    'Feature': X_logistic_df.columns,
    'Coefficient': lr_model.coef_[0]
})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
print(feature_importance.sort_values(by='Abs_Coefficient', ascending=False).to_string())
