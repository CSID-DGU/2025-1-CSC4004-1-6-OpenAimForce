#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <model_predictor.hpp> // Your class from before
#include <omp.h>

#define MAX_JOBS 16

// --- Begin utility code ---
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) out.push_back(item);
    return out;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) data.push_back(split(line, ','));
    return data;
}

struct PlayerLogInfo {
    std::string pid;
    int label;
    std::string logfile_name;
};

// You may need to adjust this to match your CSV
std::vector<PlayerLogInfo> load_metadata(const std::string& csv_path) {
    std::vector<PlayerLogInfo> info;
    auto data = read_csv(csv_path);
    int pid_col = 0, label_col = 0, log_col = 0, aim_col = 0;
    for (int i = 0; i < data[0].size(); ++i) {
        if (data[0][i] == "pid") pid_col = i;
        if (data[0][i] == "aimhack") aim_col = i;
        if (data[0][i] == "logfile_name") log_col = i;
    }
    for (int i = 1; i < data.size(); ++i) {
        PlayerLogInfo rec;
        rec.pid = data[i][pid_col];
        rec.label = (data[i][aim_col] == "1" ? 0 : 1);
        rec.logfile_name = data[i][log_col];
        info.push_back(rec);
    }
    return info;
}

std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l)) lines.push_back(l);
    return lines;
}

// --- Feature normalization (hardcode your StandardScaler means/stds for each feature) ---
static constexpr int max_seq_len = 500;
static constexpr int num_features = 8;
static constexpr float padding_value = -999.0f;
static constexpr float epsilon = 1e-9f;

// Example values: fill in with your actual values from Python scaler
float scaler_means[num_features] = {/*...*/};
float scaler_stds[num_features] = {/*...*/};

#include <vector>
#include <string>
#include <regex>
#include <cmath>
#include <algorithm>

std::vector<std::vector<float>> extract_sequences_from_log(
    const std::vector<std::string>& log_content_lines,
    const std::string& player_id,
    int max_seq_len = 500
) {
    const std::string player_prefix = "player_pid_" + player_id;
    const float epsilon = 1e-9f;
    const int num_event_features = 8;
    std::vector<std::vector<float>> player_sequence;

    struct PrevState {
        float x = NAN, y = NAN, z = NAN, yaw = NAN, pitch = NAN;
        int64_t t_microsec = -1;
        float omega_yaw = NAN, omega_pitch = NAN;
        float alpha_yaw = NAN, alpha_pitch = NAN;
    } prev_state;

    int cumulative_shots = 0, cumulative_hits = 0;

    for (size_t i = 0; i < log_content_lines.size(); ++i) {
        std::string line = log_content_lines[i];
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        std::vector<float> features(num_event_features, 0.0f);
        bool event_occurred = false;

        // Parse time
        std::smatch time_match;
        int64_t current_t = -1;
        if (std::regex_search(line, time_match, std::regex(R"(t=(\d+)\s*µs)"))) {
            current_t = std::stoll(time_match[1]);
        }

        // [MOVE] line
        if (line.find("[MOVE] " + player_prefix + ":") == 0) {
            event_occurred = true;
            std::smatch pos_match, ang_match;
            if (std::regex_search(line, pos_match, std::regex(R"(x=([\d\.-]+),\s*y=([\d\.-]+),\s*z=([\d\.-]+))")) &&
                std::regex_search(line, ang_match, std::regex(R"(yaw=(\d+),\s*pitch=(-?\d+))")) && current_t != -1) {
                float x = std::stof(pos_match[1]);
                float y = std::stof(pos_match[2]);
                float z = std::stof(pos_match[3]);
                int yaw = std::stoi(ang_match[1]);
                int pitch = std::stoi(ang_match[2]);

                if (!std::isnan(prev_state.t_microsec) && prev_state.t_microsec != -1 && current_t > prev_state.t_microsec) {
                    float dt = (current_t - prev_state.t_microsec) / 1'000'000.0f;
                    if (dt < epsilon) dt = epsilon;

                    float delta_yaw = yaw - prev_state.yaw;
                    if (!std::isnan(prev_state.yaw)) {
                        if (delta_yaw > 180) delta_yaw -= 360;
                        else if (delta_yaw < -180) delta_yaw += 360;
                    }
                    float delta_pitch = pitch - prev_state.pitch;

                    float omega_y = delta_yaw / dt;
                    float omega_p = delta_pitch / dt;
                    features[0] = omega_y;
                    features[1] = omega_p;

                    float alpha_y = 0.0f, alpha_p = 0.0f;
                    if (!std::isnan(prev_state.omega_yaw)) {
                        alpha_y = (omega_y - prev_state.omega_yaw) / dt;
                        alpha_p = (omega_p - prev_state.omega_pitch) / dt;
                        features[2] = alpha_y;
                        features[3] = alpha_p;

                        if (!std::isnan(prev_state.alpha_yaw)) {
                            features[4] = (alpha_y - prev_state.alpha_yaw) / dt;
                        }
                    }
                    features[5] = std::abs(omega_p) / (std::abs(omega_y) + epsilon);

                    prev_state.omega_yaw = omega_y;
                    prev_state.omega_pitch = omega_p;
                    prev_state.alpha_yaw = alpha_y;
                    prev_state.alpha_pitch = alpha_p;
                }
                prev_state.x = x;
                prev_state.y = y;
                prev_state.z = z;
                prev_state.yaw = yaw;
                prev_state.pitch = pitch;
            }
        }
        // [SHOOT]
        else if (line.find("[SHOOT] " + player_prefix + ":") == 0) {
            cumulative_shots++;
            event_occurred = true;
        }
        // [HIT]
        else if (line.find("[HIT] " + player_prefix + ":") == 0) {
            features[7] = 1.0f;
            cumulative_hits++;
            event_occurred = true;
        }
        // [DEAD] (own)
        else if (line.find("[DEAD] " + player_prefix + ":") == 0) {
            event_occurred = true;
        }
        // [DEAD] (general)
        else if (line.find("[DEAD]") == 0) {
            if (i + 1 < log_content_lines.size()) {
                std::string pattern = "\\[.*?\\]\\s+" + player_prefix + "\\s+\\w+\\s+\\S+";
                std::regex r(pattern);
                if (std::regex_match(log_content_lines[i + 1], r)) {
                    event_occurred = true;
                }

            }
        }

        // Only emit if something happened
        if (event_occurred) {
            features[6] = cumulative_shots > 0 ? (float)cumulative_hits / cumulative_shots : 0.0f;
            player_sequence.push_back(features);
            if (current_t != -1) prev_state.t_microsec = current_t;
        }
    }

    if (max_seq_len > 0 && player_sequence.size() > (size_t)max_seq_len) {
        player_sequence.resize(max_seq_len);
    }
    return player_sequence;
}


void normalize_sequence(std::vector<std::vector<float>>& sequence) {
    // Python: for feature indices except 7, 9, apply (x-mean)/std if not padding
    // Let's assume 7 and 9 don't exist in your 8 features, so only skip 7
    for (int f_idx = 0; f_idx < num_features; ++f_idx) {
        if (f_idx == 7 /*|| f_idx == 9*/) continue;
        for (auto& row : sequence) {
            if (row[f_idx] != padding_value) {
                row[f_idx] = (row[f_idx] - scaler_means[f_idx]) / (scaler_stds[f_idx] + epsilon);
            }
        }
    }
}

std::vector<size_t> load_indices(const std::string& filename) {
    std::vector<size_t> indices;
    std::ifstream f(filename);
    size_t idx;
    while (f >> idx) indices.push_back(idx);
    return indices;
}


// --- Main ---


int main(int argc, char** argv) {
    // Step 1: Load metadata
    auto meta = load_metadata("../../model_dev/dataset/game_db_6.csv");
    std::string log_base = "../../model_dev/dataset/id_stripped/";

    std::vector<std::vector<std::vector<float>>> all_sequences;
    std::vector<int> all_labels;
    std::mutex append_mutex;

    // Step 2: Parallel extract all sequences
    #pragma omp parallel for num_threads(MAX_JOBS) schedule(dynamic)
    for (size_t i = 0; i < meta.size(); ++i) {
        const auto& entry = meta[i];
        if (entry.logfile_name.empty()) continue;
        std::string path = log_base + "game_" + entry.logfile_name;
        auto lines = read_lines(path);
        if (lines.empty()) continue;
        auto seq = extract_sequences_from_log(lines, entry.pid);
        if (!seq.empty()) {
            std::lock_guard<std::mutex> lock(append_mutex);
            all_sequences.push_back(seq);
            all_labels.push_back(entry.label);
            std::cout << seq.size() << " entries loaded!" << std::endl;
        }
    }

    // Step 3: Normalize all sequences (no padding)
    for (auto& seq : all_sequences) {
        normalize_sequence(seq);
    }

    // Step 4: Use all data as test set
    // Load test indices
    std::vector<size_t> test_indices = load_indices("../test_indices.txt");

    // Select only those entries for X_test, y_test
    std::vector<std::vector<std::vector<float>>> X_test;
    std::vector<int> y_test;
    for (size_t i : test_indices) {
        if (i < all_sequences.size()) {
            X_test.push_back(all_sequences[i]);
            y_test.push_back(all_labels[i]);
        }
    }


    // Step 5: Predict and collect results
    ModelPredictor model("../compat_model");

    std::vector<int> pred_classes;
    std::vector<float> pred_probs;
    for (const auto& seq : X_test) {
        auto out_tensor = model.predict(seq);
        float prob = out_tensor.flat<float>()(0);
        pred_probs.push_back(prob);
        pred_classes.push_back(prob > 0.5f ? 1 : 0);
    }

    // Step 6: Metrics
    int tp=0, tn=0, fp=0, fn=0;
    for (size_t i=0; i<pred_classes.size(); ++i) {
        if (y_test[i] == 1 && pred_classes[i] == 1) tp++;
        else if (y_test[i] == 1 && pred_classes[i] == 0) fn++;
        else if (y_test[i] == 0 && pred_classes[i] == 0) tn++;
        else if (y_test[i] == 0 && pred_classes[i] == 1) fp++;
    }
    float acc = float(tp+tn) / std::max(1,int(tp+tn+fp+fn));
    float prec = tp + fp > 0 ? float(tp) / (tp+fp) : 0.0f;
    float rec = tp + fn > 0 ? float(tp) / (tp+fn) : 0.0f;
    float f1 = (prec+rec > 0) ? 2*prec*rec/(prec+rec) : 0.0f;

    // Output
    std::cout << "Accuracy: " << acc << "\n";
    std::cout << "Precision: " << prec << "\n";
    std::cout << "Recall: " << rec << "\n";
    std::cout << "F1-score: " << f1 << "\n";
    std::cout << "Confusion Matrix: [TN=" << tn << " FP=" << fp << " FN=" << fn << " TP=" << tp << "]\n";
    return 0;
}