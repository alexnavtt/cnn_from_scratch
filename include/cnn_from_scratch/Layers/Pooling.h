#pragma once

#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

enum PoolingType{
    MAX,
    MIN,
    AVG
};
    
class Pooling : public ModelLayer{
public:
    /**
     * Constructor: Set the pooling layer size, stride and type 
     */
    Pooling(dim2 dim, dim2 stride, PoolingType type = MAX);

    /**
     * Inform the caller that this layer is a Pooling layer 
     */
    ModelFlowMode getType() const override;

    /**
     * Ensure that the input size is at least as big as the pool size
     */
    bool checkSize(const SimpleMatrix<double>& input) override;

    /**
     * Set the number of affected indices to remember
     */
    void setBatchSize(size_t batch_size) override;

    /**
     * Get the appropriate output size given the input size
     * @param input_dim         The dimensions of the input data
     * @return                  The dimensions of the output data
     */
    dim3 outputSize(const dim3& input_dim) const;

    /**
     * Pool an input data point 
     */
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input, size_t batch_idx = 0) override;

    /**
     * Get the loss derivative with respect to the input for the most recent 
     * data to pass through the layer
     * @param X                 The input data that had last been passed to this layer
     * @param dLdY              The loss gradient from the previous layer
     * @return                  The loss gradient to pass to subsequent layers
     */
    SimpleMatrix<double> getdLdX(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY, size_t batch_idx);

    /**
     * Update the layer given the previous layer gradient. For a pooling layer there
     * is no data to update, and so this simply returns the loss gradient with respect
     * to the input X 
     */
    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& X,    const SimpleMatrix<double>&, 
        const SimpleMatrix<double>& dLdY, size_t batch_idx , bool) override;

    /**
     * Convert the pooling layer configuration to a standard ascii text format 
     */
    std::string serialize() const override;

    /**
     * Given an input stream holding data written by Pooling::serialize, update
     * the configuration of this layer to match the configuration in the stream 
     */
    bool deserialize(std::istream& is) override;

private:
    // Pool size
    dim2 dim_;

    // Stride
    dim2 stride_;

    // Pooling type. Can be MAX, MIN, or AVG
    PoolingType type_;

    // Stored incdices affected by the last pooling operation. Used in backpropagation
    std::vector<SimpleMatrix<dim3>> affected_indices_;
};

} // namespace my_cnn
