#pragma once

#include "cnn_from_scratch/Layers/ModelFlow.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
class ModelLayer{
public:
    std::string name;
    SimpleMatrix<double> weights;
    SimpleMatrix<double> biases;

    ModelLayer() = default;

    ModelLayer(unsigned num_outputs, dim3 weights_dim = dim3{}) :
    biases(dim3(num_outputs, 1, 1)),
    weights(weights_dim)
    {}


    virtual ModelFlowMode getType() const = 0;
    virtual bool checkSize(const SimpleMatrix<double>& input){return true;}
    virtual void setBatchSize(size_t batch_size) {}
    virtual void applyBatch(double learning_rate) {}
    virtual SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input, size_t batch_idx = 0) = 0;
    virtual SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& input, 
        const SimpleMatrix<double>& output, 
        const SimpleMatrix<double>& output_grad, 
        size_t batch_idx, 
        bool last_layer = false
    ) = 0;
    virtual std::string serialize() const = 0;
    virtual bool deserialize(std::istream& is) = 0;
};

} // namespace my_cnn
