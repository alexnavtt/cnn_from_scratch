#pragma once

#include <vector>
#include <string>
#include <memory>
#include "cnn_from_scratch/Layers/Kernel.h"
#include "cnn_from_scratch/Layers/Pooling.h"
#include "cnn_from_scratch/Layers/ModelFlow.h"
#include "cnn_from_scratch/Layers/ConnectedLayer.h"

namespace my_cnn{

template<typename OutputDataType>
struct ModelResults{
    bool knows_true_value = false;
    OutputDataType true_label;
    size_t label_idx;
    size_t true_label_idx;
    float loss = -1.0f;
    float confidence = -1.0f;
    std::vector<SimpleMatrix<float>> layer_inputs;
};

enum LossFunction{
    CROSS_ENTROPY
};

enum OutputFunction{
    UNSPECIFIED_OUTPUT,
    SOFTMAX
};

template<typename InputDataType, typename OutputDataType>
class ModelDescription{
public:
    std::vector<std::shared_ptr<ModelLayer>> layers;
    std::vector<OutputDataType> output_labels;
    LossFunction loss_function;

    bool saveModel(std::string filename);
    bool loadModel(std::string filename);
    void addKernel(dim3 size, size_t count, ModelActivationFunction activation);
    void addPooling(dim2 size, dim2 stride, PoolingType type);
    void addConnectedLayer(size_t output_size);
    void setOutputLabels(std::vector<OutputDataType> labels, OutputFunction output_function = SOFTMAX);
    ModelResults<OutputDataType> forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label = nullptr);
    void assignLoss(ModelResults<OutputDataType>& result, OutputDataType label);
    void backwardsPropagation(ModelResults<OutputDataType>& result, OutputDataType label, float learning_rate);
    float lossFcn(const SimpleMatrix<double>& probabilities, size_t true_label_idx) const;

private:
    size_t kernel_count_     = 0;
    size_t pooling_count_    = 0;
    size_t fully_conn_count_ = 0;
    OutputFunction output_function_ = UNSPECIFIED_OUTPUT;
};

} // end namespace my_cnn

#include <ModelDescription.tpp>
