#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/SimpleMatrix.h"

namespace my_cnn{
    
class Kernel {
public:
    Kernel(dim3 dim, unsigned stride) :
    weights(dim),
    stride(stride)
    {
        std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
        // Set random weights in the interval [0, 1] upon construction
        for (float& w : weights){
            w = static_cast<float>(std::rand()) / RAND_MAX;
        }
        biases.resize(dim.z, 0);
    }

    void setInputData(const SimpleMatrix<float>* input_data){
        input_data_ = input_data;
    }

    SimpleMatrix<float> convolve(){
        if (not input_data_) 
            throw std::runtime_error("No input data provided");

        const auto channel_count = weights.dim(2);
        if (input_data_->dim(2) != channel_count)
            throw std::out_of_range("Mismatched channel count for convolution");

        SimpleMatrix<float> output({
            input_data_->dim(0)-weights.dim(0)+1, 
            input_data_->dim(1)-weights.dim(1)+1, 
            channel_count
        });

        const dim3 channel_strip{1, 1, channel_count};
        dim3 idx{0, 0, 0};
        for (idx.x = 0; idx.x < (input_data_->dim(0) - weights.dim(0)); idx.x++){
            for (idx.y = 0; idx.y < (input_data_->dim(1) - weights.dim(1)); idx.y++){
                output.subMatView(idx, channel_strip) 
                    = (input_data_->subMat(idx, weights.dims()) * weights).channelSum();
            }
        }

        // Normalize the output to remain in the bounds [0, 1]
        output += output.abs().min();
        output /= output.max();

        return output;
    }

    SimpleMatrix<float> weights;
    std::valarray<float> biases;
    unsigned stride = 1;

private:
    const SimpleMatrix<float>* input_data_;
};

} // namespace my_cnn
