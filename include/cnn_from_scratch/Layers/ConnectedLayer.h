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

    /**
     * Inform the caller that this layer is a Fully Connected layer 
     */
    ModelFlowMode getType() const override;

    /**
     * Set the batch size for backpropagation so the layer knows how
     * many results to store for gradient calculation 
     */
    void setBatchSize(size_t batch_size) override;

    /**
     * Apply the stored gradients for the batch to the weight and bias matrices 
     */
    void applyBatch(double learning_rate) override;

    /**
     * Checks that the input (regardless of whether it has already been flattened)
     * contains the appropriate number of elements for forward propagation. The first
     * time that this function is called, the dimensions of the input (which should not
     * be flattened) are taken as the ground truth value for all future calls
     * TODO: Move size determination to constructor and have ModelDescription handle 
     *       figuring out what size it should be at the time of construction 
     */
    bool checkSize(const SimpleMatrix<double>& input_data) override;

    /**
     * Given an input matrix of any shape, reshape it to a column vector V and return 
     * the matrix M = W*V + B where W are the layer weights and B are the layer biases 
     */
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input_data, size_t idx = 0) override;

    /**
     * Given the input and the corresponding resulting loss gradient from the next layer, 
     * determine the loss gradient with respect to the weight matrix
     */
    SimpleMatrix<double> getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY);

    /**
     * Given the loss gradient with respect the output, determine the loss gradient with
     * respect to the input 
     */
    SimpleMatrix<double> getdLdX(const SimpleMatrix<double>& dLdY);

    /**
     * Given the loss gradient with respect the output, determine the loss gradient with
     * respect to the bias vector 
     */
    SimpleMatrix<double> getdLdB(const SimpleMatrix<double>& dLdY);

    /**
     * Update the weights and biases of the fully connected layer, and return the loss
     * gradient with respect to the input for further backpropagation in other layers
     * @param X             The input matrix from the forward pass
     * @param Y             The (unactivated) output matrix from the forward pass
     * @param dLdZ          The loss gradient with respect to the output
     * @param batch_idx     The index within a batch for this particular input/output/gradient set
     * @param last_layer    Flag indicating whether or not there are layers previous to this one.
     *                      This is a small optimization the prevents calculation of the input 
     *                      loss gradient for the final layer 
     */
    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>& Y, 
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
    dim3 input_dim_;
    bool initialized_ = false;

    // Vectors of gradient matrices for batch backpropagation
    std::vector<SimpleMatrix<double>> weight_gradients_;
    std::vector<SimpleMatrix<double>> bias_gradients_;
};

} // namespace my_cnn
