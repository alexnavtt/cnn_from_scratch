#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelLayer.h"

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
            input_data.dim(0) - dim_.x + 2*dim_.x*pad_inputs + 1,
            input_data.dim(1) - dim_.y + 2*dim_.y*pad_inputs + 1,
            (uint)biases.size()
        );
    }

    [[nodiscard]]
    static SimpleMatrix<float> convolve(const SimpleMatrix<float>& filter, const SimpleMatrix<float>& biases, const SimpleMatrix<float>& data, bool padded = false){
        // Pad the input if necessary
        SimpleMatrix<float> input = padded ? padInput(data, filter.dim()) : data;

        // Calculate the output size based on the input and filter sizes
        dim3 output_size(
            input.dim().x - filter.dim().x + 1,
            input.dim().y - filter.dim().y + 1,
            biases.size()
        );
        SimpleMatrix<float> output(output_size);

        // This is the size of the filter sub region that we'll be extracting at every step
        const dim3 sub_region_dim(filter.dim().x, filter.dim().y, data.dim().z);

        // Step through the input and accumulate the results in the output
        dim3 idx{0, 0, 0};
        for (idx.x = 0; idx.x < output_size.x; idx.x++){
            for (idx.y = 0; idx.y < output_size.y; idx.y++){
                // Extract the sub region
                SimpleMatrix<float> sub_region = input.subMatCopy(idx, sub_region_dim);

                // For each filter layer, apply the convolution process to this sub-region
                for (idx.z = 0; idx.z < biases.size(); idx.z++){
                    // Multiply by the weights and add the biases
                    output(idx) = 
                        sum(sub_region * filter.slices(idx.z*data.dim().z, data.dim().z)) + biases[idx.z];
                }
                idx.z = 0;
            }
        }

        return output;
    }

    [[nodiscard]] 
    SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input_data) override {

        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        // Create the output container
        dim3 output_size = outputSize(input_data);
        auto output = convolve(weights, biases, input_data, true);
        activate(output);
        return output;
    }

    [[nodiscard]] 
    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& input, const SimpleMatrix<float>& output_grad, float learning_rate) override {
        return input;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
    bool pad_inputs = true;
};

} // namespace my_cnn
