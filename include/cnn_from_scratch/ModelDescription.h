#pragma once

#include <vector>
#include <string>
#include <string>
#include "cpp_timer/Timer.h"
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
    cpp_timer::Timer timer;

    bool saveModel(std::string filename);
    bool loadModel(std::string filename);
    void addKernel(Kernel kernel, std::string name = "");
    void addPooling(Pooling pool, std::string name = "");
    void addConnectedLayer(size_t output_size, std::string name = "");
    void setOutputLabels(std::vector<OutputDataType> labels) {output_labels = labels;}
    result_t forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label = nullptr);
    void backwardsPropagation(const result_t& result, float learning_rate);
    float lossFcn(const SimpleMatrix<float>& probabilities, const OutputDataType& true_label) const;
};

#define TIMER_INSTANCE timer

} // end namespace my_cnn

#include <ModelDescription.tpp>
