#pragma once

#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"

namespace my_cnn{
    
class Activation : public ModelLayer{
public:
    Activation(ModelActivationFunction activation) : activation_(activation) {}

    ModelFlowMode getType() const override {
        return ACTIVATION;
    };

    bool checkSize(const SimpleMatrix<double>& input) override {
        return true;
    }

    static double sigmoid(double f) {
        return 1.0/(1.0 + std::exp(-f));
    }

    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input) override {
        switch (activation_){
            case RELU:
                return apply(input, [](double f){return f > 0 ? f : 0;});

            case SIGMOID:
                return apply(input, [](double f){return sigmoid(f);});

            default:
            case LINEAR:
                return input;

            case TANGENT:
                return apply(input, (double(*)(double))std::tanh);
                
            case LEAKY_RELU:
                return apply(input, [](double f){return std::max(0.1f*f, f);});
        }
    }

    auto activationGradient(const SimpleMatrix<double>& activated_output) const {
        using result_t = UnaryOperationResult<const SimpleMatrix<double>&, double(*)(const SimpleMatrix<double>&, const dim3&)>;
        
        switch (activation_){
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

    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>&, const SimpleMatrix<double>& Z, 
        const SimpleMatrix<double>& dLdZ, double, bool) override
    {
        return dLdZ * activationGradient(Z);
    }

    std::string serialize() const override {
        std::stringstream ss;
        ss << "Activation\n";
        ss << toString(activation_) << "\n";
        return ss.str();
    }

    bool deserialize(std::istream& is) override {
        serialization::expect<void>(is, "Activation");

        std::string activation_string;
        std::getline(is, activation_string);
        activation_ = fromString(activation_string);
        return true;
    }

protected:
    ModelActivationFunction activation_ = LINEAR;
};

} // namespace my_cnn
