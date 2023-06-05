#pragma once

#include "cnn_from_scratch/Layers/ModelLayer.h"

namespace my_cnn{

class Softmax : public ModelLayer{
public:
    Softmax(size_t output_label_count) : 
    output_label_count_(output_label_count)
    {
        tag = OUTPUT;
    }    

    bool checkSize(const SimpleMatrix<double>& input) override {
        return (input.isColumn() || input.isRow()) && 
               (input.size() == output_label_count_);
    }

    void assignTrueLabel(size_t true_label_idx){
        if (true_label_idx >= output_label_count_){
            std::stringstream ss;
            ss << "Cannot assign true label of " << true_label_idx << " when there are only " << output_label_count_ << " labels";
            throw std::out_of_range(ss.str());
        }

        true_label_idx_ = true_label_idx;
        knows_true_label_ = true;
    }

    SimpleMatrix<double> softMax(const SimpleMatrix<double>& X){
        // Offset the data by the max to keep numerical feasibility
        const double max_val = my_cnn::max(X);
        const auto  offset_X = X - max_val;

        // Apply the softmax formula
        const SimpleMatrix<double> exp_X = my_cnn::exp(offset_X);
        return exp_X/my_cnn::sum(exp_X);
    }

    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input) override {
        if (not checkSize(input))
            throw ModelLayerException("Softmax input size is ill formed");

        return softMax(input);   
    }

    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& input, const SimpleMatrix<double>& output, 
        const SimpleMatrix<double>&, double, double) override 
    {
        if (not knows_true_label_){
            throw ModelLayerException("Cannot backpropagate softmax layer without knowledge of true label");
        }

        SimpleMatrix<double> gradient = output;
        gradient[true_label_idx_] -= 1;
        return gradient;
    }

private:
    bool knows_true_label_ = false;
    size_t true_label_idx_;
    size_t output_label_count_;
};
    
} // namespace my_cnn
