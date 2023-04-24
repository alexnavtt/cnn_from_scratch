#pragma once

#include <vector>
#include <string>
#include <memory>
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/ModelFlow.h"
#include "cnn_from_scratch/ConnectedLayer.h"

namespace my_cnn{

template<typename OutputDataType>
struct ModelResults{
    size_t label_idx;
    size_t true_label_idx;
    float loss = -1.0f;
    std::vector<SimpleMatrix<float>> layer_inputs;
};

enum LossFunction{
    CROSS_ENTROPY
};

template<typename InputDataType, typename OutputDataType>
class ModelDescription{
public:
    using result_t = ModelResults<OutputDataType>;
    std::vector<std::shared_ptr<ModelLayer>> layers;
    std::vector<OutputDataType> output_labels;
    ModelFlow flow;
    LossFunction loss_function;

    bool saveModel(std::string filename);
    bool loadModel(std::string filename);
    void addKernel(std::shared_ptr<Kernel> kernel, std::string name = "");
    void addPooling(std::shared_ptr<Pooling> pool, std::string name = "");
    void addConnectedLayer(size_t output_size, std::string name = "");
    void setOutputLabels(std::vector<OutputDataType> labels) {output_labels = labels;}
    result_t forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label = nullptr);
    void backwardsPropagation(const result_t& result, float learning_rate);
    float lossFcn(const SimpleMatrix<float>& probabilities, const OutputDataType& true_label) const;
    SimpleMatrix<float> softMax(const SimpleMatrix<float>& X);
};

} // end namespace my_cnn

#include <ModelDescription.tpp>
