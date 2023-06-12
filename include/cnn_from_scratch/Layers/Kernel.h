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
    Kernel(dim3 filter_dim, unsigned num_filters_in, unsigned stride) :
    ModelLayer(num_filters_in, {filter_dim.x, filter_dim.y, filter_dim.z*num_filters_in}),
    dim_(filter_dim),
    stride(stride),
    num_filters(num_filters_in)
    {
        std::srand(std::chrono::steady_clock::now().time_since_epoch().count());
        // Set random weights in the interval [0, 1] upon construction
        for (double& w : weights){
            w = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
        }
        for (double& b : biases){
            b = 1 - 2*static_cast<double>(std::rand()) / RAND_MAX;
        }

        // Normalize weights and biases so they start on stable footing
        weights /= l2Norm(weights);
        biases /= l2Norm(biases);
    }

    ModelFlowMode getType() const override {
        return KERNEL;
    }

    template<typename MatrixType>
    static SimpleMatrix<double> padInput(MatrixType&& input_data, const dim3 filter_dim){
        // Create the augmented input data
        SimpleMatrix<double> padded({
            input_data.dim().x + 2*(filter_dim.x - 1),
            input_data.dim().y + 2*(filter_dim.y - 1),
            input_data.dim().z
        });

        padded.subMatView({filter_dim.x-1, filter_dim.y-1, 0}, input_data.dim()) = std::forward<MatrixType>(input_data);
        return padded;
    }

    bool checkSize(const SimpleMatrix<double>& input_data) override {
        return input_data.dim(2) == dim_.z;
    }

    dim3 outputSize(const SimpleMatrix<double>& input_data) const{
        return dim3(
            input_data.dim(0) - dim_.x + 1,
            input_data.dim(1) - dim_.y + 1,
            num_filters
        );
    }


    [[nodiscard]] 
    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input_data) override {
        // Make sure the input size is what we expect based on weights and biases dimensions
        if (not checkSize(input_data))
            throw ModelLayerException("Mismatched channel count for convolution");

        SimpleMatrix<double> output(outputSize(input_data));
        
        // Convolve the input with the weights and add the biases
        for (size_t i = 0; i < num_filters; i++){
            const auto filter = weights.slices(i*dim_.z, dim_.z);
            for (size_t j = 0; j < dim_.z; j++){
                auto _ = global_timer.scopedTic("Convolution");
                const auto W = filter.slice(j);
                const auto I = input_data.slice(j);
                output.slice(i) += convolve(I, W, dim2(stride, stride));
            }
            output.slice(i) += biases[i];
        }

        // Return
        return output;
    }

    [[nodiscard]]
    SimpleMatrix<double> getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
        STIC;
        const int num_channels = dim_.z;

        // Create the output matrix
        SimpleMatrix<double> dLdW(weights.dim());

        // Loop through all of the filters and find the gradient for each one
        for (int filter_idx = 0; filter_idx < num_filters; filter_idx++){

            // Get references to the data just for this filter
            SubMatrixView<const double> output_gradient = dLdY.slice(filter_idx);
            SubMatrixView<double>       weight_gradient = dLdW.slices(filter_idx*num_channels, num_channels);

            for (int channel_idx = 0; channel_idx < num_channels; channel_idx++){
                // Get the input channel that corresponds to this filter channel
                SubMatrixView<const double> input_channel = X.slice(channel_idx);

                // Calculate the derivate of the loss with respect to the weights for this channel
                weight_gradient.slice(channel_idx) = convolve(input_channel, output_gradient, {1, 1});
            }
        }

        return dLdW;
    }

    [[nodiscard]]
    SimpleMatrix<double> getdLdX(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
        STIC;
        const int num_channels = dim_.z;

        // Create the output matrix
        SimpleMatrix<double> dLdX(X.dim());

        // Loop through all of the filters and find the gradient for each one
        for (int filter_idx = 0; filter_idx < num_filters; filter_idx++){
            // Get references to the data just for this filter
            const SubMatrixView<double> filter          = weights.slices(filter_idx*num_channels, num_channels);
            SubMatrixView<const double> output_gradient = dLdY.slice(filter_idx);

            // Create storage for the input gradient corresponding to this filter
            SimpleMatrix<double> filter_gradient(dLdX.dim());

            // Loop through each channel of the input and calculate the gradient
            for (int channel_idx = 0; channel_idx < num_channels; channel_idx++){
                const auto filter_channel   = filter.slice(channel_idx);
                const auto rotated_gradient = rotate<2>(output_gradient);
                const auto padded_filter    = padInput(filter_channel, output_gradient.dim());

                TIC("convolution");
                filter_gradient.slice(channel_idx) = convolve(padded_filter, rotated_gradient, {1, 1});
                TOC("convolution");
            }

            dLdX = dLdX + filter_gradient;
        }

        return dLdX;
    }

    [[nodiscard]]
    SimpleMatrix<double> getdLdB(const SimpleMatrix<double>& B, const SimpleMatrix<double>& dLdY){
        STIC;
        
        SimpleMatrix<double> dLdB(B.dim());
        for (int filter_idx = 0; filter_idx < num_filters; filter_idx++){
            dLdB[filter_idx] = sum(dLdY.slice(filter_idx));
        }

        return dLdB;
    }

    [[nodiscard]] 
    SimpleMatrix<double> propagateBackward(
            const SimpleMatrix<double>& X, const SimpleMatrix<double>& Z, 
            const SimpleMatrix<double>& dLdZ, double learning_rate, bool last_layer) 
        override 
    {
        // SimpleMatrix<double> dLdY = getdLdY(Z, dLdZ);
        weights -= learning_rate * getdLdW(X, dLdZ);       
        biases  -= learning_rate * getdLdB(biases, dLdZ);
        return last_layer ? SimpleMatrix<double>() : getdLdX(X, dLdZ);
    }

    std::string serialize() const override {
        std::stringstream ss;
        ss << "Kernel\n";
        serialization::place(ss, dim_.x, "x");
        serialization::place(ss, dim_.y, "y");
        serialization::place(ss, dim_.z, "z");
        serialization::place(ss, num_filters, "n");
        ss << "weights\n";
        weights.serialize(ss);
        ss << "biases\n";
        biases.serialize(ss);
        return ss.str();
    }

    bool deserialize(std::istream& is) override {

        // First line should be just the word "Kernel"
        serialization::expect<void>(is, "Kernel");

        // Get the dimension of the kernel
        dim_.x = serialization::expect<int>(is, "x");
        dim_.y = serialization::expect<int>(is, "y");
        dim_.z = serialization::expect<int>(is, "z");

        // Get the number of kernels
        num_filters = serialization::expect<unsigned>(is, "n");

        // Read the weights matrix
        serialization::expect<void>(is, "weights");
        if (not weights.deserialize(is)) return false;

        // Read the biases matrix
        serialization::expect<void>(is, "biases");
        if (not biases.deserialize(is)) return false;

        return true;
    }

private:
    dim3 dim_;

public:
    unsigned stride = 1;
    unsigned num_filters = 0;
    ModelFlowMode type = KERNEL;
};

} // namespace my_cnn
