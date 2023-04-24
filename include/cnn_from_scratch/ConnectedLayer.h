#pragma once

#include <assert.h>
#include <valarray>
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/timerConfig.h"
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

    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& badly_shaped_X, const SimpleMatrix<float>& Y, const SimpleMatrix<float>& dLdY, float learning_rate) override{        
        // Need to reshape the input appropriately
        SimpleMatrix<float> X = badly_shaped_X;
        X.reshape(badly_shaped_X.size(), 1, 1);

        TIC("activationGradient");
        const SimpleMatrix<float> dLdz = dLdY * activationGradient(Y);
        TOC("activationGradient");
        TIC("updateWeights");
        weights -= learning_rate * matrixMultiply(dLdz, transpose(X));
        TOC("updateWeights");
        TIC("updateBiases");
        biases -= learning_rate * dLdz;
        TOC("updateBiases");

        // We need to reshape the gradient to what the previous layer would be expecting
        TIC("outputGradient");
        SimpleMatrix<float> dLdX = matrixMultiply(transpose(weights), dLdz);
        dLdX.reshape(badly_shaped_X.dim());
        TOC("outputGradient");
        return dLdX;
    }
};

} // namespace my_cnn
