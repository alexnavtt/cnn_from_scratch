#pragma once

#include <valarray>
#include <cnn_from_scratch/SimpleMatrix.h>

namespace my_cnn{

enum ModelActivationFunction{
    RELU,
    SIGMOID,
    LINEAR,
    TANGENT,
    LEAKY_RELU
};
    
class ModelLayer{
public:
    SimpleMatrix<float> weights;
    std::valarray<float> biases;
    bool initialized = false;
    ModelActivationFunction activation = LINEAR;

    ModelLayer() = default;

    ModelLayer(unsigned num_outputs, dim3 weights_dim = dim3{}) :
    biases(num_outputs),
    weights(weights_dim)
    {}

    void activate(SimpleMatrix<float>& output_data) const{
        switch (activation){
            case RELU:
                std::for_each(std::begin(output_data), std::end(output_data), [](float& f){f = f > 0 ? f : 0;});
                break;

            case SIGMOID:
                std::for_each(std::begin(output_data), std::end(output_data), [](float& f){f = 1.0f/(1 + expf(f));});
                break;

            default:
            case LINEAR:
                break;

            case TANGENT:
                output_data = std::tanh(output_data);
                break;
                
            case LEAKY_RELU:
                std::for_each(std::begin(output_data), std::end(output_data), [](float& f){f = std::max(0.1f*f, f);});
                break;
        }
    }

    virtual bool checkSize(const SimpleMatrix<float>& input) = 0;
    virtual SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input) = 0;
    virtual SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& input, const SimpleMatrix<float>& output_grad, float learning_rate) = 0;
};

} // namespace my_cnn
