#include "cnn_from_scratch/Layers/Kernel.h"

namespace my_cnn{

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

Kernel::Kernel(dim3 filter_dim, unsigned num_filters_in, unsigned stride) :
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

ModelFlowMode Kernel::getType() const {
    return KERNEL;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void Kernel::setBatchSize(size_t batch_size) {
    weight_gradients_.clear();
    bias_gradients_.clear();

    weight_gradients_.resize(batch_size);
    bias_gradients_.resize(batch_size);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void Kernel::applyBatch(double learning_rate){
    SimpleMatrix<double> avg_weight_grad(weights.dim(), 0.0);
    SimpleMatrix<double> avg_bias_grad(biases.dim(), 0.0);

    // Accumulate the stored gradientss
    for (size_t i = 0; i < weight_gradients_.size(); i++){
        avg_weight_grad += weight_gradients_[i]; 
        avg_bias_grad   += bias_gradients_[i];  
    }

    // Average the gradients
    avg_weight_grad /= weight_gradients_.size();
    avg_bias_grad   /= bias_gradients_.size();

    // Apply
    weights -= learning_rate * avg_weight_grad;
    biases  -= learning_rate * avg_bias_grad;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

template<typename MatrixType>
SimpleMatrix<double> Kernel::padInput(MatrixType&& input_data, const dim3 filter_dim){
    // Create the augmented input data
    SimpleMatrix<double> padded({
        input_data.dim().x + 2*(filter_dim.x - 1),
        input_data.dim().y + 2*(filter_dim.y - 1),
        input_data.dim().z
    });

    padded.subMatView({filter_dim.x-1, filter_dim.y-1, 0}, input_data.dim()) = std::forward<MatrixType>(input_data);
    return padded;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool Kernel::checkSize(const SimpleMatrix<double>& input_data) {
    return input_data.dim(2) == dim_.z;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

dim3 Kernel::outputSize(const SimpleMatrix<double>& input_data) const{
    return dim3(
        input_data.dim(0) - dim_.x + 1,
        input_data.dim(1) - dim_.y + 1,
        num_filters
    );
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Kernel::propagateForward(SimpleMatrix<double>&& input_data, size_t) {
    // Make sure the input size is what we expect based on weights and biases dimensions
    if (!checkSize(input_data))
        throw ModelLayerException("Mismatched channel count for convolution");

    SimpleMatrix<double> output(outputSize(input_data));
    
    // Convolve the input with the weights and add the biases
    for (size_t i = 0; i < num_filters; i++){
        const auto filter = weights.slices(i*dim_.z, dim_.z);
        for (size_t j = 0; j < dim_.z; j++){
            // stic("Convolution");
            const auto W = filter.slice(j);
            const auto I = input_data.slice(j);
            output.slice(i) += convolve(I, W, dim2(stride, stride));
        }
        output.slice(i) += biases[i];
    }

    // Return
    return output;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Kernel::getdLdW(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
    // STIC;
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Kernel::getdLdX(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
    // STIC;
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

            filter_gradient.slice(channel_idx) = convolve(padded_filter, rotated_gradient, {1, 1});
        }

        dLdX = dLdX + filter_gradient;
    }

    return dLdX;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Kernel::getdLdB(const SimpleMatrix<double>& dLdY){
    // STIC;
    
    SimpleMatrix<double> dLdB(biases.dim());
    for (int filter_idx = 0; filter_idx < num_filters; filter_idx++){
        dLdB[filter_idx] = sum(dLdY.slice(filter_idx));
    }

    return dLdB;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
    
SimpleMatrix<double> Kernel::propagateBackward(
        const SimpleMatrix<double>& X, const SimpleMatrix<double>& Z, 
        const SimpleMatrix<double>& dLdZ, size_t batch_idx, bool last_layer) 
{
    // SimpleMatrix<double> dLdY = getdLdY(Z, dLdZ);
    weight_gradients_[batch_idx] = getdLdW(X, dLdZ);
    bias_gradients_[batch_idx]   = getdLdB(dLdZ);
    // weights -= learning_rate * getdLdW(X, dLdZ);       
    // biases  -= learning_rate * getdLdB(dLdZ);
    return last_layer ? SimpleMatrix<double>() : getdLdX(X, dLdZ);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

std::string Kernel::serialize() const {
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool Kernel::deserialize(std::istream& is) {

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
    if (!weights.deserialize(is)) return false;

    // Read the biases matrix
    serialization::expect<void>(is, "biases");
    if (!biases.deserialize(is)) return false;

    return true;
}

} // namespace my_cnn
