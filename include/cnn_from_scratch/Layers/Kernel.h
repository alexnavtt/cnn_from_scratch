#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/Layers/ModelFlow.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"
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
        for (double& w : weights){
            w = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
        }
        for (double& b : biases){
            b = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
        }

        // Normalize weights and biases so they start on stable footing
        weights /= l2Norm(weights);
        biases /= l2Norm(biases);
    }

    template<typename MatrixType>
    static SimpleMatrix<double> padInput(MatrixType&& input_data, const dim3 filter_dim){
        // Create the augmented input data
        SimpleMatrix<double> padded({
            input_data.dim().x + 2*(filter_dim.x - 1),
            input_data.dim().y + 2*(filter_dim.y - 1),
            input_data.dim().z
        });

        padded.subMatView({filter_dim.x-1, filter_dim.y-1, 0}, input_data.dim()) = std::forward<MatrixType>(input_data);
        return padded;
    }

    bool checkSize(const SimpleMatrix<double>& input_data) override {
        return input_data.dim(2) == dim_.z;
    }

    dim3 outputSize(const SimpleMatrix<double>& input_data) const{
        return dim3(
            input_data.dim(0) - dim_.x + 1,
            input_data.dim(1) - dim_.y + 1,
            num_filters
        );
    }


    [[nodiscard]] 
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input_data) override {
        // Make sure the input size is what we expect based on weights and biases dimensions
        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        SimpleMatrix<double> output(outputSize(input_data));
        
        // Convolve the input with the weights and add the biases
        for (size_t i = 0; i < num_filters; i++){
            const auto filter = weights.slices(i*dim_.z, dim_.z);
            for (size_t j = 0; j < dim_.z; j++){
                auto _ = global_timer.scopedTic("Convolution");
                const auto W = filter.slice(j);
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
    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>& Y, 
            const SimpleMatrix<double>& dLdY, double learning_rate, double norm_penalty) 
        override 
        {
        // Apply the gradient from the activation layer to get the output gradient
        const SimpleMatrix<double> dLdZ = dLdY * activationGradient(Y);
        const dim2 stride(1,1);
        const double bias_norm = l2Norm(biases);

        SimpleMatrix<double> dLdX(X.dim());
        for (size_t i = 0; i < num_filters; i++){
            // Extract the part of the gradient salient to this filter
            SubMatrixView<const double> gradient_layer = dLdZ.slice(i);
            SubMatrixView<double> filter = weights.slices(i*dim_.z, dim_.z);

            // Update output gradient
            TIC("inputGradient");
            for (size_t j = 0; j < dim_.z; j++){
                SubMatrixView<double> filter_layer = filter.slice(j);
                SubMatrixView<double> input_gradient = dLdX.slice(j);
                SimpleMatrix<double> padded_weights = padInput(rotate<2>(filter_layer), gradient_layer.dim());

                input_gradient += convolve(padded_weights, gradient_layer, stride);
            }
            TOC("inputGradient");

            // Update weights
            TIC("updateWeights");
            for (size_t j = 0; j < dim_.z; j++){
                SubMatrixView<const double> input_layer = X.slice(j);
                SubMatrixView<double> filter_layer = filter.slice(j);
                filter_layer -= learning_rate * convolve(input_layer, gradient_layer, stride);
            }
            TOC("updateWeights");

            // Update biases
            TIC("updateBiases");
            biases[i] -= learning_rate * sum(gradient_layer);
            TOC("updateBiases");
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