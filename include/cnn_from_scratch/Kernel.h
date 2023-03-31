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

    SimpleMatrix<float> apply(dim3 idx, size_t channel){
        // Create a submatrix of the input data for the region we want to filter
        SimpleMatrix<float> submatrix = input_data_->subMatrix(idx, weights.dims());
        // Element-wise weight multiplication
        submatrix *= weights;
        // Bias addition
        submatrix += biases[channel];
    }

    SimpleMatrix<float> weights;
    std::valarray<float> biases;
    unsigned stride = 1;

private:
    const SimpleMatrix<float>* input_data_;
};

} // namespace my_cnn
