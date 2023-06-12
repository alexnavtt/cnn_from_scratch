#pragma once

#include <valarray>
#include "cnn_from_scratch/Layers/ModelFlow.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

enum ModelActivationFunction{
    RELU,
    SIGMOID,
    LINEAR,
    TANGENT,
    LEAKY_RELU
};

static inline std::string toString(ModelActivationFunction f){
    switch (f){
        case RELU:          return "RELU";
        case SIGMOID:       return "SIGMOID";
        case LINEAR:        return "LINEAR";
        case TANGENT:       return "TANGENT";
        case LEAKY_RELU:    return "LEAKY_RELU";
        default: 
            throw std::runtime_error("Unknown ModelActivationFunction");
    }
}

static inline ModelActivationFunction fromString(const std::string& s){
    if (s == "RELU")        return RELU;
    if (s == "SIGMOID")     return SIGMOID;
    if (s == "LINEAR")      return LINEAR;
    if (s == "TANGENT")     return TANGENT;
    if (s == "LEAKY_RELU")  return LEAKY_RELU;
    else
        throw std::runtime_error("Unknown string \"" + s + "\" for ModelActivationFunction");
}
    
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
    virtual bool checkSize(const SimpleMatrix<double>& input) = 0;
    virtual SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input) = 0;
    virtual SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& input, 
        const SimpleMatrix<double>& output, 
        const SimpleMatrix<double>& output_grad, 
        double learning_rate, 
        bool last_layer = false
    ) = 0;
    virtual std::string serialize() const = 0;
    virtual bool deserialize(std::istream& is) = 0;
};

} // namespace my_cnn
