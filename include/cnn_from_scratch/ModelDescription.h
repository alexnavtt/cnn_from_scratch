#pragma once

#include <vector>
#include <string>
#include <string_view>
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/ModelFlow.h"
#include "cnn_from_scratch/ConnectedLayer.h"

namespace my_cnn{

template<typename OutputDataType>
struct ModelResults{
    OutputDataType label;
    size_t label_idx;
    float loss;
    std::vector<SimpleMatrix<float>> layer_inputs;
};

enum LossFunction{
    CROSS_ENTROPY
};

template<typename InputDataType, typename OutputDataType>
class ModelDescription{
public:
    using result_t = ModelResults<OutputDataType>;
    std::vector<Kernel> kernels;
    std::vector<Pooling> pools;
    std::vector<ConnectedLayer> connected_layers;
    std::vector<OutputDataType> output_labels;
    ModelFlow flow;
    LossFunction loss_function;

    bool saveModel(std::string filename);
    bool loadModel(std::string filename);
    void addKernel(Kernel kernel, std::string_view name = "");
    void addPooling(Pooling pool, std::string_view name = "");
    void addConnectedLayer(size_t output_size, std::string_view name = "");
    void setOutputLabels(std::vector<OutputDataType> labels) {output_labels = labels;}
    result_t forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label = nullptr);
    void backwardsPropagation(const result_t& result, float learning_rate);
    float lossFcn(const SimpleMatrix<float>& probabilities, const OutputDataType& true_label) const;
};

} // end namespace my_cnn

#include <ModelDescription.tpp>
