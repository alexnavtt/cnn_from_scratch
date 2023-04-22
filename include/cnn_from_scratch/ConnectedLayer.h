#pragma once

#include <assert.h>
#include <valarray>
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
class ConnectedLayer : public ModelLayer{
public:
    using ModelLayer::ModelLayer;

    bool checkSize(const SimpleMatrix<float>& input_data) override{
        // If this is the first time at this layer, resize and apply random values
        dim3 expected_size(biases.size(), input_data.size(), 1);
        if (not initialized){
            initialized = true;
            weights = SimpleMatrix<float>(expected_size);
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

    SimpleMatrix<float> propagateForward(SimpleMatrix<float>&& input_data) override{
        // The input size is always a column vector
        input_data.reshape(dim3(input_data.size(), 1, 1));

        if (not checkSize(input_data)){
            throw ModelLayerException("Invalid input size for fully connected layer. Input has size " + 
                std::to_string(input_data.size()) + " and this layer has size " + std::to_string(weights.dim(0)));
        }

        SimpleMatrix<float> output = matrixMultiply(weights, input_data) + biases;
        activate(output);
        return output;
    }

    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& input_data, const SimpleMatrix<float>& output, const SimpleMatrix<float>& output_grad, float learning_rate) override{
        weights -= learning_rate * matrixMultiply(output_grad, transpose(input_data));
        biases  -= learning_rate * output_grad;
        return matrixMultiply(transpose(weights), output_grad);
    }
};

} // namespace my_cnn
