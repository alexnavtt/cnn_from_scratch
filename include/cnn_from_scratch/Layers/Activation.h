#pragma once

#include <array>
#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"

namespace my_cnn{

// All activation functions supported by this library
enum ModelActivationFunction{
    RELU        = 0,
    SIGMOID     = 1,
    LINEAR      = 2,
    TANGENT     = 3,
    LEAKY_RELU  = 4
};

// Enum to string conversion
static inline std::string toString(ModelActivationFunction f){
    static const std::array<std::string, 5> activation_strings{
        "RELU",
        "SIGMOID",
        "LINEAR",
        "TANGENT",
        "LEAKY_RELU"
    };

    try{
        return activation_strings.at(f);
    }catch(std::out_of_range){
        throw std::runtime_error("Unknown ModelActivationFunction: " + std::to_string((int)f));
    }
}

// String to enum conversion
static inline ModelActivationFunction fromString(const std::string& s){
    if (s == "RELU")        return RELU;
    if (s == "SIGMOID")     return SIGMOID;
    if (s == "LINEAR")      return LINEAR;
    if (s == "TANGENT")     return TANGENT;
    if (s == "LEAKY_RELU")  return LEAKY_RELU;
    else
        throw std::runtime_error("Unknown string \"" + s + "\" for ModelActivationFunction");
}

// Actual activation functions themselves
double reluFcn(double val);
double sigmoidFcn(double val);
double linearFcn(double val);
double tangentFcn(double val);
double leakyRelyFcn(double val);

double reluGradFcn(double val);
double sigmoidGradFcn(double val);
double linearGradFcn(double val);
double tangentGradFcn(double val);
double leakyRelyGradFcn(double val);
    
// Network layer responsible for handling activations
class Activation : public ModelLayer{
public:
    /**
     * Constructor: Sets activation function 
     */
    Activation(ModelActivationFunction activation);

    /**
     * Inform the caller that this layer is an Activation layer 
     */
    ModelFlowMode getType() const override;

    /**
     * Return the activated form of the given input matrix 
     */
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input) override;

    /**
     * Return the gradient of the activation function for the given activated matrix 
     */
    SimpleMatrix<double> activationGradient(const SimpleMatrix<double>& activated_output) const;

    /**
     * Given the loss gradient from the previous layer, return the loss gradient
     * to be propagated to subsequent layers 
     */
    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>&, const SimpleMatrix<double>& Z, 
        const SimpleMatrix<double>& dLdZ, double, bool) override;

    /**
     * Convert the layer configuration to a standard ascii text format 
     */
    std::string serialize() const override;

    /**
     * Given an input stream holding data written by serialize, update the
     * configuration of this layer to match the configuration in the stream 
     */
    bool deserialize(std::istream& is) override;

protected:
    ModelActivationFunction activation_;
    double(*activationFunction_)(double);
    double(*activationGradientFunction_)(double);
};

} // namespace my_cnn
