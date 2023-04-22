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
    Kernel(dim3 filter_dim, unsigned num_filters, unsigned stride) :
    ModelLayer(num_filters, {filter_dim.x, filter_dim.y, filter_dim.z*num_filters}),
    dim_(filter_dim),
    stride(stride)
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

    static SimpleMatrix<float> padInput(const SimpleMatrix<float>& input_data, const dim3 filter_dim){
        auto _ = global_timer.scopedTic("padInput");
        // Create the augmented input data
        SimpleMatrix<float> padded({
            input_data.dim(0) + 2*(filter_dim.x - 1),
            input_data.dim(1) + 2*(filter_dim.y - 1),
            input_data.dim(2)
        });

        padded.subMatView({filter_dim.x-1, filter_dim.y-1, 0}, input_data.dims()) = input_data;
        return padded;
    }

    bool checkSize(const SimpleMatrix<float>& input_data) override {
        return input_data.dim(2) == dim_.z;
    }

    dim3 outputSize(const SimpleMatrix<float>& input_data) const{
        return dim3(
            input_data.dim(0) - dim_.x + 1,
            input_data.dim(1) - dim_.y + 1,
            (uint)biases.size()
        );
    }

    [[nodiscard]]
    static SimpleMatrix<float> convolve(const SimpleMatrix<float>& filter, const SimpleMatrix<float>& data, bool padded = false){
        auto _ = global_timer.scopedTic("convolve");
        // Pad the input if necessary
        SimpleMatrix<float> input = padded ? padInput(data, filter.dim()) : data;

        // Calculate the output size based on the input and filter sizes
        dim3 output_size(
            input.dim().x - filter.dim().x + 1,
            input.dim().y - filter.dim().y + 1,
            filter.dim().z / data.dim().z
        );
        SimpleMatrix<float> output(output_size);

        // This is the size of the filter sub region that we'll be extracting at every step
        const dim3 sub_region_dim(filter.dim().x, filter.dim().y, data.dim().z);

        // Step through the input and accumulate the results in the output
        dim3 idx{0, 0, 0};
        for (uint x = 0; x < output_size.x; x++){
            for (uint y = 0; y < output_size.y; y++){
                // Extract the sub region
                auto sub_region = input.subMatView({x, y, 0}, sub_region_dim);
                auto _ = global_timer.scopedTic("calculateOutputValue");

                // For each filter layer, apply the convolution process to this sub-region
                for (uint z = 0; z < output_size.z; z++){
                    // Multiply by the weights and add the biases
                    output(x, y, z) = sum(sub_region * filter.slices(z*data.dim().z, data.dim().z));
                }
            }
        }

        return output;
    }

    [[nodiscard]] 
    SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input_data) override {
        // Make sure the input size is what we expect based on weights and biases dimensions
        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        // Convolve the input with the weights
        auto output = convolve(weights, input_data, false);

        // Add the bias terms
        global_timer.tic("addBiases");
        for (size_t i = 0; i < output.dim().z; i++){
            output.slice(i) += biases[i];
        }
        global_timer.toc("addBiases");

        // Apply the activation function
        activate(output);

        // Return
        return output;
    }

    [[nodiscard]] 
    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& X, const SimpleMatrix<float>& Y, const SimpleMatrix<float>& dLdY, float learning_rate) override {
        const SimpleMatrix<float> dLdZ = dLdY * activationGradient(dLdY);
        // Update the weights
        weights -= learning_rate * convolve(dLdZ, X);
        // Update the biases
        for (size_t i = 0; i < biases.size(); i++){
            biases[i] -= learning_rate * sum(dLdZ.slice(i));
        }
        // Calculate the input gradient
        return X;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
};

} // namespace my_cnn
