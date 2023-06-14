#pragma once

#include <random>
#include <chrono>
#include "cnn_from_scratch/Layers/ModelFlow.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/Serialization.h"

#define debug(x) std::cout << #x << x

namespace my_cnn{

class Kernel : public ModelLayer{
public:
    /**
     * Constructor: Set the filter dimensions, filter count, and stride 
     */
    Kernel(dim3 filter_dim, unsigned num_filters_in, unsigned stride);

    /**
     * Inform the caller that this layer is a Kernel layer 
     */
    ModelFlowMode getType() const override;

    /**
     * Clear and resize the storage for weight and bias gradients 
     */
    void setBatchSize(size_t batch_size) override;

    /**
     * Update the weights and biases based on the batch-stored gradients 
     */
    void applyBatch(double learning_rate) override;

    /**
     * Given an input matrix and filter dimension, return a version of the input matrix that
     * is padded to allow a full convolution between the filter and the padded input 
     */
    template<typename MatrixType>
    static SimpleMatrix<double> padInput(MatrixType&& input_data, const dim3 filter_dim);

    /**
     * Ensure that the depth of the input matches the depth of the kernel 
     */
    bool checkSize(const SimpleMatrix<double>& input_data) override;

    /**
     * Given the dimensions of the input data, return the dimensions of the ouptput data 
     */
    dim3 outputSize(const SimpleMatrix<double>& input_data) const;

    /**
     * Pass the given input matrix through a convolutional filter and return the result  
     */
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input_data, size_t idx = 0) override;

    /**
     * Given the input and the loss gradient with respect to the output, get the loss
     * gradient with respect to the weight matrix 
     */
    SimpleMatrix<double> getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY);

    /**
     * Given the input and the loss gradient with respect to the output, get the loss
     * gradient with respect to the input to pass to subsequent layers 
     */
    SimpleMatrix<double> getdLdX(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY);

    /**
     * Given the loss matrix with respect to the output, get the loss gradient with
     * respect to the bias vector
     */
    SimpleMatrix<double> getdLdB(const SimpleMatrix<double>& dLdY);

    /**
     * Update the weights and biases of the kernel, and return the loss gradient
     * with respect to the input for further backpropagation in other layers
     * @param X             The input matrix from the forward pass
     * @param Y             The (unactivated) output matrix from the forward pass
     * @param dLdZ          The loss gradient with respect to the output
     * @param learning_rate The step size with which to change the weight matrices
     * @param last_layer    Flag indicating whether or not there are layers previous to this one.
     *                      This is a small optimization the prevents calculation of the input 
     *                      loss gradient for the final layer 
     */
    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>& Z, 
            const SimpleMatrix<double>& dLdZ, size_t batch_idx, bool last_layer) override;

    /**
     * Convert the layer configuration to a standard ascii text format 
     */
    std::string serialize() const override;

    /**
     * Given an input stream holding data written by serialize, update the
     * configuration of this layer to match the configuration in the stream 
     */
    bool deserialize(std::istream& is) override;

private:
    dim3 dim_;

    // Vectors of gradient matrices for batch backpropagation
    std::vector<SimpleMatrix<double>> weight_gradients_;
    std::vector<SimpleMatrix<double>> bias_gradients_;

public:
    unsigned stride = 1;
    unsigned num_filters = 0;
    ModelFlowMode type = KERNEL;
};

} // namespace my_cnn
