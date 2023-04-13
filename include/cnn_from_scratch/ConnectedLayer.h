#pragma once

#include <assert.h>
#include <valarray>
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/SimpleMatrix.h"

namespace my_cnn{
    
class ConnectedLayer : public ModelLayer{
public:
    using ModelLayer::ModelLayer;

    bool checkSize(const SimpleMatrix<float>& input_data) override{
        // If this is the first time at this layer, resize and apply random values
        dim3 expected_size(biases.size(), input_data.size(), 1);
        if (not initialized){
            initialized = true;
            weights.resize(expected_size);
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
            return ( weights.dims() == expected_size );
        }
    }

    SimpleMatrix<float> apply(const SimpleMatrix<float>& input_data) override{
        if (not checkSize(input_data)){
            throw ModelLayerException("Invalid input size for fully connected layer. Input has size " + 
                std::to_string(input_data.size()) + " and this layer has size " + std::to_string(weights.dim(0)));
        }

        SimpleMatrix<float> output = weights.matMul(input_data) + biases;
        activate(output);
        return output;
    }
};

} // namespace my_cnn
