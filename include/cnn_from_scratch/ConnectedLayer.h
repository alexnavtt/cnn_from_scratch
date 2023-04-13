#pragma once

#include <valarray>
#include <cnn_from_scratch/SimpleMatrix.h>

namespace my_cnn{
    
struct ConnectedLayer{
    SimpleMatrix<float> weights;
    std::valarray<float> biases;
    bool initialized = false;
};

} // namespace my_cnn
