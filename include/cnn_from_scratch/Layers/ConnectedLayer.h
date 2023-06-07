#pragma once

#include <assert.h>
#include <valarray>
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
class ConnectedLayer : public ModelLayer{
public:
    using ModelLayer::ModelLayer;

    bool checkSize(const SimpleMatrix<double>& input_data) override{
        // If this is the first time at this layer, resize and apply random values
        dim3 expected_size(biases.size(), input_data.size(), 1);
        if (not initialized_){
            initialized_ = true;
            std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
            // Set random weights in the interval [0, 1] upon construction
            weights = SimpleMatrix<double>(expected_size);
            for (double& w : weights){
                w = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
            }
            for (double& b : biases){
                b = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
            }

            // Normalize weights and biases so they start on stable footing
            weights /= l2Norm(weights);
            biases /= l2Norm(biases);

            return true;
        }
        // Otherwise check to make sure the size is correct
        else{
            return ( weights.dims() == expected_size );
        }
    }

    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input_data) override{
        // The input size is always a column vector
        input_data.reshape(dim3(input_data.size(), 1, 1));

        if (not checkSize(input_data)){
            throw ModelLayerException("Invalid input size for fully connected layer. Input has size " + 
                std::to_string(input_data.size()) + " and this layer has size " + std::to_string(weights.dim(0)));
        }

        SimpleMatrix<double> output = matrixMultiply(weights, input_data) + biases;
        activate(output);
        return output;
    }

    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& badly_shaped_X, const SimpleMatrix<double>& Z, 
            const SimpleMatrix<double>& dLdZ, double learning_rate, bool last_layer) 
    override
    {        
        // Need to reshape the input appropriately
        SimpleMatrix<double> X = badly_shaped_X;
        X.reshape(badly_shaped_X.size(), 1, 1);

        TIC("activationGradient");
        const SimpleMatrix<double> dLdY = dLdZ * activationGradient(Z);
        TOC("activationGradient");
        TIC("updateWeights");
        weights -= learning_rate * matrixMultiply(dLdY, transpose(X));
        TOC("updateWeights");
        TIC("updateBiases");
        biases -= learning_rate * dLdY;
        TOC("updateBiases");

        // We need to reshape the gradient to what the previous layer would be expecting
        SimpleMatrix<double> dLdX;
        if (not last_layer){
            stic("outputGradient");
            dLdX = matrixMultiply(transpose(weights), dLdY);
            dLdX.reshape(badly_shaped_X.dim());
        }
        return dLdX;
    }

private:
    bool initialized_ = false;
};

} // namespace my_cnn
