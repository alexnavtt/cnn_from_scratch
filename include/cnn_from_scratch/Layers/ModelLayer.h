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
    
class ModelLayer{
public:
    std::string name;
    SimpleMatrix<double> weights;
    SimpleMatrix<double> biases;
    ModelActivationFunction activation = LINEAR;

    ModelLayer() = default;

    ModelLayer(unsigned num_outputs, dim3 weights_dim = dim3{}) :
    biases(dim3(num_outputs, 1, 1)),
    weights(weights_dim)
    {}

    static double sigmoid(double f) {
        return 1.0/(1.0 + std::exp(-f));
    }

    void activate(SimpleMatrix<double>& output_data) const{
        switch (activation){
            case RELU:
                modify(output_data, [](double f){return f > 0 ? f : 0;});
                break;

            case SIGMOID:
                modify(output_data, [](double f){return sigmoid(f);});
                break;

            default:
            case LINEAR:
                break;

            case TANGENT:
                modify(output_data, (double(*)(double))std::tanh);
                break;
                
            case LEAKY_RELU:
                modify(output_data, [](double f){return std::max(0.1f*f, f);});
                break;
        }
    }

    auto activationGradient(const SimpleMatrix<double>& activated_output) const {
        using result_t = UnaryOperationResult<const SimpleMatrix<double>&, double(*)(const SimpleMatrix<double>&, const dim3&)>;
        
        switch (activation){
            case RELU:
                return result_t(activated_output, 
                    [](const SimpleMatrix<double>& M, const dim3& idx){
                        return (double)(M(idx) > 0);
                    }
                );

            case SIGMOID:
                return result_t(activated_output, 
                    [](const SimpleMatrix<double>& M, const dim3& idx){
                        const auto val = M(idx);
                        return sigmoid(val)*(1 - sigmoid(val));
                    }
                );

            default:
            case LINEAR:
                return result_t(activated_output, 
                    [](const SimpleMatrix<double>& M, const dim3& idx){
                        return 1.0;
                    }
                );

            case TANGENT:
                return result_t(activated_output, 
                    [](const SimpleMatrix<double>& M, const dim3& idx){
                        const auto val = M(idx);
                        return 1.0f - std::tanh(val)*std::tanh(val);
                    }
                );

            case LEAKY_RELU:
                return result_t(activated_output, 
                    [](const SimpleMatrix<double>& M, const dim3& idx){
                        return M(idx) > 0 ? 1.0 : 0.1;
                    }
                );
        }
    }

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
