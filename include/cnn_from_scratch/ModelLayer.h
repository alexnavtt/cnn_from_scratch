#pragma once

#include <valarray>
#include <cnn_from_scratch/Matrix/SimpleMatrix.h>

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
    SimpleMatrix<float> biases;
    bool initialized = false;
    ModelActivationFunction activation = LINEAR;

    ModelLayer() = default;

    ModelLayer(unsigned num_outputs, dim3 weights_dim = dim3{}) :
    biases(dim3(num_outputs, 1, 1)),
    weights(weights_dim)
    {}

    void activate(SimpleMatrix<float>& output_data) const{
        switch (activation){
            case RELU:
                modify(output_data, [](float f){return f > 0 ? f : 0;});
                break;

            case SIGMOID:
                modify(output_data, [](float f){return 1.0f/(1 + expf(f));});
                break;

            default:
            case LINEAR:
                break;

            case TANGENT:
                modify(output_data, (float(*)(float))std::tanh);
                break;
                
            case LEAKY_RELU:
                modify(output_data, [](float f){return std::max(0.1f*f, f);});
                break;
        }
    }

    virtual bool checkSize(const SimpleMatrix<float>& input) = 0;
    virtual SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input) = 0;
    virtual SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& input, const SimpleMatrix<float>& output_grad, float learning_rate) = 0;
};

} // namespace my_cnn
