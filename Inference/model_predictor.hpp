#pragma once
#if defined(_WIN32)
  #define MODEL_PREDICTOR_API __declspec(dllexport)
#else
  #define MODEL_PREDICTOR_API __attribute__((visibility("default")))
#endif

#include <string>
#include <vector>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

class MODEL_PREDICTOR_API ModelPredictor {
public:
    explicit ModelPredictor(const std::string& model_dir);
    tensorflow::Tensor predict(const std::vector<std::vector<float>>& sequence);
private:
    tensorflow::SavedModelBundleLite bundle_;
    std::string input_name_;
    std::string output_name_;
    int max_seq_len_;
    int num_features_;
    float pad_value_;
    void init_signature_names();
};
