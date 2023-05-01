#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelFlow.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/timerConfig.h"

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

        // Normalize weights and biases so they start on stable footing
        weights /= l2Norm(weights);
        biases /= l2Norm(biases);
    }

    template<typename MatrixType>
    static SimpleMatrix<float> padInput(MatrixType&& input_data, const dim3 filter_dim){
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
    SimpleMatrix<float> propagateBackward(
            const SimpleMatrix<float>& X, const SimpleMatrix<float>& Y, 
            const SimpleMatrix<float>& dLdY, float learning_rate, float norm_penalty) 
        override 
        {
        // Apply the gradient from the activation layer to get the output gradient
        const SimpleMatrix<float> dLdZ = dLdY * activationGradient(Y);
        const dim2 stride(1,1);
        const float bias_norm = l2Norm(biases);

        SimpleMatrix<float> dLdX(X.dim());
        for (size_t i = 0; i < num_filters; i++){
            // Extract the part of the gradient salient to this filter
            SubMatrixView<const float> gradient_layer = dLdZ.slice(i);
            SubMatrixView<float> filter = weights.slices(i*dim_.z, dim_.z);
            const float filter_norm = l2Norm(filter);

            // Update weights
            TIC("updateWeights");
            for (size_t j = 0; j < dim_.z; j++){
                SubMatrixView<const float> input_layer = X.slice(j);
                SubMatrixView<float> filter_layer = filter.slice(j);
                filter_layer -= learning_rate * convolve(input_layer, gradient_layer, stride);
            }
            TOC("updateWeights");

            // Update biases
            TIC("updateBiases");
            biases[i] -= learning_rate * sum(gradient_layer);
            TOC("updateBiases");

            // Update output gradient
            TIC("inputGradient");
            for (size_t j = 0; j < dim_.z; j++){
                SubMatrixView<float> filter_layer = filter.slice(j);
                SubMatrixView<float> input_gradient = dLdX.slice(j);
                SimpleMatrix<float> padded_weights = padInput(rotate<2>(filter_layer), gradient_layer.dim());

                input_gradient += convolve(padded_weights, gradient_layer, stride);
            }
            TOC("inputGradient");
        }

        // Apply the norm penalty
        weights *= (1 - norm_penalty);

        return dLdX;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
    unsigned num_filters = 0;
    ModelFlowMode type = KERNEL;
};

} // namespace my_cnn
