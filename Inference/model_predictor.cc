#include "model_predictor.hpp"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include <stdexcept>
#include <iostream>

ModelPredictor::ModelPredictor(const std::string& model_dir)
    : max_seq_len_(500), num_features_(8), pad_value_(-999.0f)
{
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    auto status = tensorflow::LoadSavedModel(session_options, run_options, model_dir, {"serve"}, &bundle_);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load model: " + status.ToString());
    }
    init_signature_names();
}

void ModelPredictor::init_signature_names() {
    // You might need to inspect or hardcode these based on your model
    // Here, we assume "serving_default" and typical Keras naming conventions
    auto sigs = bundle_.GetSignatures();
    if (sigs.find("serving_default") != sigs.end()) {
        const auto& def = sigs.at("serving_default");
        // Usually one input and one output
        if (def.inputs_size() == 1 && def.outputs_size() == 1) {
            input_name_ = def.inputs().begin()->second.name();
            output_name_ = def.outputs().begin()->second.name();
        } else {
            throw std::runtime_error("SignatureDef has unexpected number of inputs/outputs.");
        }
    } else {
        throw std::runtime_error("No 'serving_default' signature found.");
    }
}

tensorflow::Tensor ModelPredictor::predict(const std::vector<std::vector<float>>& sequence) {
    // Preprocess: pad or truncate sequence to max_seq_len_, fill with pad_value_
    std::vector<std::vector<float>> padded(max_seq_len_, std::vector<float>(num_features_, pad_value_));
    size_t seq_len = std::min(sequence.size(), static_cast<size_t>(max_seq_len_));
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < num_features_; ++j) {
            padded[i][j] = sequence[i][j];
        }
    }

    // Make a Tensor of shape [1, max_seq_len, num_features]
    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({1, max_seq_len_, num_features_})
    );

    auto tensor_mapped = input_tensor.tensor<float, 3>();
    for (int i = 0; i < max_seq_len_; ++i) {
        for (int j = 0; j < num_features_; ++j) {
            tensor_mapped(0, i, j) = padded[i][j];
        }
    }

    // Call session
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {input_name_, input_tensor}
    };
    std::vector<tensorflow::Tensor> outputs;
    auto run_status = bundle_.GetSession()->Run(inputs, {output_name_}, {}, &outputs);
    if (!run_status.ok()) {
        throw std::runtime_error("Inference failed: " + run_status.ToString());
    }

    return outputs[0]; // Could extract float from outputs[0] depending on output shape
}
