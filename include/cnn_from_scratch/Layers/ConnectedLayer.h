#pragma once

#include <assert.h>
#include <valarray>
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
class ConnectedLayer : public ModelLayer{
public:
    using ModelLayer::ModelLayer;

    ModelFlowMode getType() const override {
        return FULLY_CONNECTED;
    }

    bool checkSize(const SimpleMatrix<double>& input_data) override{
        // If this is the first time at this layer, resize and apply random values
        dim3 expected_size(biases.size(), input_data.size(), 1);
        if (not initialized_){
            initialized_ = true;
            input_dim_ = input_data.dim();
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
        if (not checkSize(input_data)){
            throw ModelLayerException("Invalid input size for fully connected layer. Input has size " + 
                std::to_string(input_data.size()) + " and this layer has size " + std::to_string(weights.dim(0)));
        }

        // The input size is always a column vector
        input_data.reshape(input_data.size(), 1, 1);

        SimpleMatrix<double> output = matrixMultiply(weights, input_data) + biases;
        return output;
    }

    SimpleMatrix<double> getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
        STIC;
        SimpleMatrix<double> flatX = X;
        flatX.reshape(X.size(), 1, 1);
        return matrixMultiply(dLdY, transpose(flatX));
    }

    SimpleMatrix<double> getdLdX(const SimpleMatrix<double>& dLdY){
        STIC;
        SimpleMatrix<double> dLdX = matrixMultiply(transpose(weights), dLdY);
        dLdX.reshape(input_dim_);
        return dLdX;
    }

    SimpleMatrix<double> getdLdB(const SimpleMatrix<double>& dLdY){
        return dLdY;
    }

    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>& Z, 
            const SimpleMatrix<double>& dLdZ, double learning_rate, bool last_layer) 
    override
    {        
        weights -= learning_rate * getdLdW(X, dLdZ);
        biases  -= learning_rate * getdLdB(dLdZ);

        // We need to reshape the gradient to what the previous layer would be expecting
        SimpleMatrix<double> dLdX = last_layer ? SimpleMatrix<double>() : getdLdX(dLdZ);
        return dLdX;
    }

    std::string serialize() const override {
        std::stringstream ss;
        ss << "Connected Layer\n";
        serialization::place(ss, weights.dim().x, "x");
        serialization::place(ss, weights.dim().y, "y");
        ss << "weights\n";
        weights.serialize(ss);
        ss << "biases\n";
        biases.serialize(ss);
        return ss.str();
    }

    bool deserialize(std::istream& is) override {
        serialization::expect<void>(is, "Connected Layer");
        dim2 stream_dim;
        stream_dim.x = serialization::expect<int>(is, "x");
        stream_dim.y = serialization::expect<int>(is, "y");
        serialization::expect<void>(is, "weights");
        if (not weights.deserialize(is)) return false;
        serialization::expect<void>(is, "biases");
        if (not biases.deserialize(is)) return false;
        initialized_ = true;
        return true;
    }

private:
    dim3 input_dim_;
    bool initialized_ = false;
};

} // namespace my_cnn
