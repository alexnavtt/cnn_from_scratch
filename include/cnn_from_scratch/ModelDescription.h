#pragma once

#include <vector>
#include <string>
#include <string_view>
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/ModelFlow.h"

namespace my_cnn{

template<typename InputDataType>
class ModelDescription{
public:
    std::vector<Kernel> kernels;
    std::vector<Pooling> pools;
    std::vector<SimpleMatrix<float>> connected_layers;
    ModelFlow flow;

    bool saveModel(std::string filename);
    bool loadModel(std::string filename);
    void addKernel(Kernel kernel, std::string_view name = "");
    void addPooling(Pooling pool, std::string_view name = "");
    void addConnectedLayer(std::string_view name = "");
    void run(SimpleMatrix<InputDataType> input);
};

} // end namespace my_cnn
