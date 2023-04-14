#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/SimpleMatrix.h"
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

    SimpleMatrix<float> padInput(const SimpleMatrix<float>& input_data) const{
        // Create the augmented input data
        SimpleMatrix<float> padded({
            input_data.dim(0) + 2*dim_.x - 1,
            input_data.dim(1) + 2*dim_.y - 1,
            input_data.dim(2)
        });

        padded.subMatView({dim_.x, dim_.y, 0}, input_data.dims()) = input_data;
        return padded;
    }

    bool checkSize(const SimpleMatrix<float>& input_data) override {
        return input_data.dim(2) == dim_.z;
    }

    [[nodiscard]] SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input_data) override {

        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        SimpleMatrix<float> input_augmented = pad_inputs ? padInput(input_data) : input_data;

        // Create the output container
        SimpleMatrix<float> output({
            input_augmented.dim(0) - dim_.x + 1, 
            input_augmented.dim(1) - dim_.y + 1, 
            (uint)biases.size()
        });

        dim3 idx{0, 0, 0};
        for (idx.x = 0; idx.x < (input_augmented.dim(0) - weights.dim(0)); idx.x++){
            for (idx.y = 0; idx.y < (input_augmented.dim(1) - weights.dim(1)); idx.y++){
                // Extract the sub region
                SimpleMatrix<float> sub_region = input_augmented.subMatCopy(idx, dim_);

                // For each filter layer, apply the convolution process to this sub-region
                for (uint filter_layer = 0; filter_layer < biases.size(); filter_layer++){
                    // Get the index for the weights that need to be applied to this layer
                    auto layer_weight_idx = weights.slices(filter_layer*dim_.z, dim_.z);
                    // Multiply by the weights
                    SimpleMatrix<float> layer_sub_region = sub_region;
                    layer_sub_region *= weights[layer_weight_idx];
                    // Sum the resulting matrices and add the biases
                    float z_val = layer_sub_region.sum() + biases[filter_layer];
                    // Apply the activation function
                    // Set the value in the output layer
                    output(idx.x, idx.y, filter_layer) = z_val;
                }
            }
        }

        activate(output);
        return output;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
    bool pad_inputs = true;
};

} // namespace my_cnn
