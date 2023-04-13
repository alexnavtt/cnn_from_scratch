#pragma once

#include <assert.h>
#include <valarray>
#include <cnn_from_scratch/SimpleMatrix.h>

namespace my_cnn{
    
struct ConnectedLayer{
    SimpleMatrix<float> weights;
    std::valarray<float> biases;
    bool initialized = false;

    bool checkSize(const SimpleMatrix<float>& input_data){
        if (not initialized){
            initialized = true;
            weights.resize(input_data.size(), biases.size(), 1);
            for (auto& v : weights){
                v = (float)rand() / (float)RAND_MAX;
            }
            for (auto& v : biases){
                v = (float)rand() / (float)RAND_MAX;
            }
            return true;
        }
        // Otherwise check to make sure the size is correct
        else{
            assert(weights.dims() == dim3(input_data.size(), biases.size(), 1));
        }
        return true;
    }
};

} // namespace my_cnn
