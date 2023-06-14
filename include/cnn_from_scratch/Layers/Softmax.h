#pragma once

#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/Layers/ModelLayer.h"

namespace my_cnn{

class Softmax : public ModelLayer{
public:
    /**
     * Constructor: set number of output labels 
     */
    Softmax(size_t output_label_count);

    /**
     * Inform the caller that this layer is an output layer 
     */
    ModelFlowMode getType() const override; 

    /**
     * Confirm that the input is a 1D vector with the number of elements
     * equal to the number of output labels for the network
     */
    bool checkSize(const SimpleMatrix<double>& input) override;

    /**
     * Inform the layer which output label is correct. Used in backpropagation
     */
    void assignTrueLabel(size_t true_label_idx);

    /**
     * The actual softmax function 
     */
    SimpleMatrix<double> softMax(const SimpleMatrix<double>& X);

    /**
     * Return the final output vector after applying the softMax function
     */
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input, size_t) override;

    /**
     * Given the output and the true label, return the loss gradient with respect to the input 
     */
    SimpleMatrix<double> getdLdX(const SimpleMatrix<double> output, size_t true_label_idx);

    /**
     * Given the output and the true label, return the loss gradient with respect to the input 
     */
    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& input, const SimpleMatrix<double>& output, 
        const SimpleMatrix<double>&, size_t, bool) override;

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
    bool knows_true_label_ = false;
    size_t true_label_idx_;
    size_t output_label_count_;
};
    
} // namespace my_cnn
