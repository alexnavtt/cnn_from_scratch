#include "cnn_from_scratch/Layers/ConnectedLayer.h"

namespace my_cnn{

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

ModelFlowMode ConnectedLayer::getType() const {
    return FULLY_CONNECTED;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void ConnectedLayer::setBatchSize(size_t batch_size) {
    weight_gradients_.clear();
    bias_gradients_.clear();

    weight_gradients_.resize(batch_size);
    bias_gradients_.resize(batch_size);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void ConnectedLayer::applyBatch(double learning_rate){
    SimpleMatrix<double> avg_weight_grad(weights.dim(), 0.0);
    SimpleMatrix<double> avg_bias_grad(biases.dim(), 0.0);

    // Accumulate the stored gradientss
    for (size_t i = 0; i < weight_gradients_.size(); i++){
        avg_weight_grad += weight_gradients_[i]; 
        avg_bias_grad += bias_gradients_[i];  
    }

    // Average the gradients
    avg_weight_grad /= weight_gradients_.size();
    avg_bias_grad /= bias_gradients_.size();

    // Apply
    weights -= learning_rate * avg_weight_grad;
    biases  -= learning_rate * avg_bias_grad;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool ConnectedLayer::checkSize(const SimpleMatrix<double>& input_data){
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> ConnectedLayer::propagateForward(SimpleMatrix<double>&& input_data, size_t) {
    if (not checkSize(input_data)){
        throw ModelLayerException("Invalid input size for fully connected layer. Input has size " + 
            std::to_string(input_data.size()) + " and this layer has size " + std::to_string(weights.dim(1)));
    }

    // The input size is always a column vector
    input_data.reshape(input_data.size(), 1, 1);

    SimpleMatrix<double> output = matrixMultiply(weights, input_data) + biases;
    return output;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> ConnectedLayer::getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
    // STIC;
    SimpleMatrix<double> flatX = X;
    flatX.reshape(X.size(), 1, 1);
    return matrixMultiply(dLdY, transpose(flatX));
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> ConnectedLayer::getdLdX(const SimpleMatrix<double>& dLdY){
    // STIC;
    SimpleMatrix<double> dLdX = matrixMultiply(transpose(weights), dLdY);
    dLdX.reshape(input_dim_);
    return dLdX;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> ConnectedLayer::getdLdB(const SimpleMatrix<double>& dLdY){
    return dLdY;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> ConnectedLayer::propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>&, 
            const SimpleMatrix<double>& dLdZ, size_t batch_idx, bool last_layer)
{        
    weight_gradients_[batch_idx] = getdLdW(X, dLdZ);
    bias_gradients_[batch_idx]   = getdLdB(dLdZ);
    // weights -= learning_rate * getdLdW(X, dLdZ);
    // biases  -= learning_rate * getdLdB(dLdZ);

    // We need to reshape the gradient to what the previous layer would be expecting
    SimpleMatrix<double> dLdX = last_layer ? SimpleMatrix<double>() : getdLdX(dLdZ);
    return dLdX;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

std::string ConnectedLayer::serialize() const {
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool ConnectedLayer::deserialize(std::istream& is) {
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

    
} // namespace my_cnn
