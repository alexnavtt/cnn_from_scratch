#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"

namespace my_cnn{
    
class Kernel {
public:
    Kernel(dim3 dim, unsigned stride) :
    dim_(dim),
    weights(dim),
    biases_(dim),
    stride(stride)
    {
        std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
        // Set random weights in the interval [0, 1] upon construction
        for (float& w : weights){
            w = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    void setBias(unsigned channel, float val){
        biases_[biases_.slice(channel)] = val;
    }

    float getBias(unsigned channel){
        return biases_[dim_.x*dim_.y*channel];
    }

    void setInputData(const SimpleMatrix<float>* input_data){
        input_data_ = input_data;
    }

    SimpleMatrix<float> padInput() const{
        // Create the augmented input data
        SimpleMatrix<float> padded({
            input_data_->dim(0) + 2*weights.dim(0) - 1,
            input_data_->dim(1) + 2*weights.dim(1) - 1,
            weights.dim(2)
        });

        padded[padded.subMatIdx({weights.dim(0), weights.dim(1), 0}, input_data_->dims())] = *input_data_;
        return padded;
    }

    SimpleMatrix<float> convolve(){
        if (not input_data_) 
            throw std::runtime_error("No input data provided");

        const auto channel_count = weights.dim(2);
        if (input_data_->dim(2) != channel_count)
            throw std::out_of_range("Mismatched channel count for convolution");

        SimpleMatrix<float> input_augmented = padInput();

        // Create the output container
        SimpleMatrix<float> output({
            input_augmented.dim(0) - weights.dim(0)+1, 
            input_augmented.dim(1) - weights.dim(1)+1, 
            channel_count
        });

        const dim3 channel_strip{1, 1, channel_count};
        dim3 idx{0, 0, 0};
        for (idx.x = 0; idx.x < (input_augmented.dim(0) - weights.dim(0)); idx.x++){
            for (idx.y = 0; idx.y < (input_augmented.dim(1) - weights.dim(1)); idx.y++){
                // Extract the sub region
                SimpleMatrix<float> sub_region = input_augmented.subMatCopy(idx, weights.dims());
                // Multiply by weights (all channels at once)
                sub_region *= weights;
                // Add the biases
                sub_region += biases_;
                // Set the region in the output layer
                output[output.subMatIdx(idx, channel_strip)] = sub_region.channelSum();
            }
        }

        return output;
    }

    SimpleMatrix<float> weights;
    unsigned stride = 1;

private:
    dim3 dim_;
    SimpleMatrix<float> biases_;
    const SimpleMatrix<float>* input_data_;
};

} // namespace my_cnn
