#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"

namespace my_cnn{

enum ModelActivationFunction{
    RELU,
    SIGMOID,
    LINEAR,
    TANGENT,
    LEAKY_RELU
};
    
class Kernel {
public:
    Kernel(dim3 filter_dim, unsigned num_filters, unsigned stride) :
    dim_(filter_dim),
    num_filters_(num_filters),
    weights({dim_.x, dim_.y, dim_.z*num_filters}),
    biases(num_filters),
    stride(stride)
    {
        std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
        // Set random weights in the interval [0, 1] upon construction
        for (float& w : weights){
            w = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    void setInputData(const SimpleMatrix<float>* input_data){
        input_data_ = input_data;
    }

    SimpleMatrix<float> padInput() const{
        // Create the augmented input data
        SimpleMatrix<float> padded({
            input_data_->dim(0) + 2*dim_.x - 1,
            input_data_->dim(1) + 2*dim_.y - 1,
            input_data_->dim(2)
        });

        padded.subMatView({dim_.x, dim_.y, 0}, input_data_->dims()) = *input_data_;
        return padded;
    }

    [[nodiscard]] static float activate(const float val, ModelActivationFunction type = RELU) noexcept {
        switch (type){
            default:
            case RELU:
                return (val < 0 ? 0 : val);

            case SIGMOID:
                return 1.0f/(1 + expf(val));

            case LINEAR:
                return val;

            case TANGENT:
                return tanh(val);
                
            case LEAKY_RELU:
                return std::max(0.1f*val, val);
        }
    }

    [[nodiscard]] SimpleMatrix<float> convolve(){
        if (not input_data_) 
            throw std::runtime_error("No input data provided");

        const auto input_channel_count = input_data_->dim(2);
        if (input_channel_count != dim_.z)
            throw std::out_of_range("Mismatched channel count for convolution");

        SimpleMatrix<float> input_augmented = pad_inputs ? padInput() : *input_data_;

        // Create the output container
        SimpleMatrix<float> output({
            input_augmented.dim(0) - dim_.x + 1, 
            input_augmented.dim(1) - dim_.y + 1, 
            num_filters_
        });

        const dim3 sub_region_size(dim_.x, dim_.y, input_channel_count);
        dim3 idx{0, 0, 0};
        for (idx.x = 0; idx.x < (input_augmented.dim(0) - weights.dim(0)); idx.x++){
            for (idx.y = 0; idx.y < (input_augmented.dim(1) - weights.dim(1)); idx.y++){
                // Extract the sub region
                SimpleMatrix<float> sub_region = input_augmented.subMatCopy(idx, sub_region_size);

                // For each filter layer, apply the convolution process to this sub-region
                for (uint filter_layer = 0; filter_layer < num_filters_; filter_layer++){
                    // Get the index for the weights that need to be applied to this layer
                    auto layer_weight_idx = weights.slices(filter_layer*dim_.z, dim_.z);
                    // Multiply by the weights
                    SimpleMatrix<float> layer_sub_region = sub_region;
                    layer_sub_region *= weights[layer_weight_idx];
                    // Sum the resulting matrices and add the biases
                    float z_val = layer_sub_region.sum() + biases[filter_layer];
                    // Apply the activation function
                    z_val = activate(z_val, RELU);
                    // Set the value in the output layer
                    output(idx.x, idx.y, filter_layer) = z_val;
                }
            }
        }

        return output;
    }

private:
    dim3 dim_;
    unsigned num_filters_ = 0;
    const SimpleMatrix<float>* input_data_;

public:
    SimpleMatrix<float> weights;
    std::vector<float> biases;
    unsigned stride = 1;
    bool pad_inputs = true;
};

} // namespace my_cnn
