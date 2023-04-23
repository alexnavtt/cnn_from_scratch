#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cpp_timer/Timer.h"

extern cpp_timer::Timer global_timer;

namespace my_cnn{

class Kernel : public ModelLayer{
public:
    Kernel(dim3 filter_dim, unsigned num_filters_in, unsigned stride) :
    ModelLayer(num_filters_in, {filter_dim.x, filter_dim.y, filter_dim.z*num_filters_in}),
    dim_(filter_dim),
    stride(stride),
    num_filters(num_filters_in)
    {
        std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
        // Set random weights in the interval [0, 1] upon construction
        for (float& w : weights){
            w = 1 - 2*static_cast<float>(std::rand()) / RAND_MAX;
        }
        for (float& b : biases){
            b = 1 - 2*static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    template<typename MatrixType>
    static SimpleMatrix<float> padInput(MatrixType&& input_data, const dim3 filter_dim){
        auto _ = global_timer.scopedTic("padInput");
        // Create the augmented input data
        SimpleMatrix<float> padded({
            input_data.dim().x + 2*(filter_dim.x - 1),
            input_data.dim().y + 2*(filter_dim.y - 1),
            input_data.dim().z
        });

        padded.subMatView({filter_dim.x-1, filter_dim.y-1, 0}, input_data.dim()) = std::forward<MatrixType>(input_data);
        return padded;
    }

    bool checkSize(const SimpleMatrix<float>& input_data) override {
        return input_data.dim(2) == dim_.z;
    }

    dim3 outputSize(const SimpleMatrix<float>& input_data) const{
        return dim3(
            input_data.dim(0) - dim_.x + 1,
            input_data.dim(1) - dim_.y + 1,
            num_filters
        );
    }


    [[nodiscard]] 
    SimpleMatrix<float> propagateForward(SimpleMatrix<float>&& input_data) override {
        // Make sure the input size is what we expect based on weights and biases dimensions
        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        // Convolve the input with the weights and add the biases
        SimpleMatrix<float> output(outputSize(input_data));
        for (size_t i = 0; i < num_filters; i++){
            for (size_t j = 0; j < dim_.z; j++){
                auto _ = global_timer.scopedTic("Convolution");
                const auto W = weights.slice(i*dim_.z  + j);
                const auto I = input_data.slice(j);
                output.slice(i) += convolve(I, W, dim2(stride, stride));
            }
            output.slice(i) += biases[i];
        }

        // Apply the activation function
        activate(output);

        // Return
        return output;
    }

    [[nodiscard]] 
    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& X, const SimpleMatrix<float>& Y, const SimpleMatrix<float>& dLdY, float learning_rate) override {
        auto _ = global_timer.scopedTic("kernelBackprop");
        // Apply the gradient from the activation layer to get the output gradient
        const SimpleMatrix<float> dLdZ = dLdY * activationGradient(dLdY);

        SimpleMatrix<float> dLdX(X.dim());
        for (size_t i = 0; i < num_filters; i++){
            // Update weights
            global_timer.tic("updateWeights");
            auto gradient_layer = dLdZ.slice(i);
            for (size_t j = 0; j < dim_.z; j++){
                auto input_layer = X.slice(j);
                weights.slice(i*dim_.z + j) -= learning_rate * convolve(input_layer, gradient_layer, dim2(1, 1)); 
            }
            global_timer.toc("updateWeights");

            // Update biases
            global_timer.tic("updateBiases");
            biases[i] -= learning_rate * sum(gradient_layer);
            global_timer.toc("updateBiases");

            // Update output gradient
            global_timer.tic("outputGradient");
            for (size_t j = 0; j < dim_.z; j++){
                SimpleMatrix<float> padded_weights = padInput(rotate<2>(weights.slice(i*dim_.z + j)), gradient_layer.dim());
                global_timer.tic("convolve");
                dLdX.slice(j) += convolve(padded_weights, gradient_layer, dim2(1, 1));
                global_timer.toc("convolve");
            }
            global_timer.toc("outputGradient");
        }
        return dLdX;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
    unsigned num_filters = 0;
};

} // namespace my_cnn
