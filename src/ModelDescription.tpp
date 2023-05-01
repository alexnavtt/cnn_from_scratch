#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/ModelDescription.h"

extern cpp_timer::Timer global_timer;

namespace my_cnn{

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addKernel
    (dim3 size, size_t count, ModelActivationFunction activation)
{
    auto& K = layers.emplace_back(new Kernel(size, count, 1));
    K->activation = activation;
    K->name = "Kernel_" + std::to_string(++kernel_count_);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addPooling
    (dim2 size, dim2 stride, PoolingType type)
{
    auto& P = layers.emplace_back(new Pooling(size, stride, type));
    P->name = "Pooling_" + std::to_string(++pooling_count_);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addConnectedLayer(size_t output_size)
{
    auto& C = layers.emplace_back(new ConnectedLayer(output_size));
    C->name = "FullyConnected_" + std::to_string(++fully_conn_count_);
}

template<typename InputDataType, typename OutputDataType>
float ModelDescription<InputDataType, OutputDataType>::lossFcn(const SimpleMatrix<float>& probabilities, const OutputDataType& true_label) const{
    constexpr float eps = 1e-5;
    
    switch (loss_function){
        default:
        case CROSS_ENTROPY:
            // Find the index of the true label
            size_t true_idx = std::distance(std::begin(output_labels), std::find(std::begin(output_labels), std::end(output_labels), true_label));
            return -1*log2(probabilities[true_idx] + eps);
    }
}

template<typename InputDataType, typename OutputDataType>
SimpleMatrix<float> ModelDescription<InputDataType, OutputDataType>::softMax(const SimpleMatrix<float>& X){
    SimpleMatrix<float> out = my_cnn::exp(X - my_cnn::max(X));
    out /= my_cnn::sum(out);
    return out;
}

template<typename InputDataType, typename OutputDataType>
ModelResults<OutputDataType> ModelDescription<InputDataType, OutputDataType>::forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label)
{
    STIC;
    // If necessary, convert the input data type to float
    SimpleMatrix<float> active_data = std::move(input);

    // std::cout << "Input image is \n";
    // for (uint layer = 0; layer < active_data.dim(2); layer++){
    //     printImage(active_data.slice(layer));
    // }

    // Create the output struct and store the input
    ModelResults<OutputDataType> result;
    result.layer_inputs.reserve(layers.size() + 1);

    // For each stage of the model, apply the necessary step
    for (size_t i = 0; i < layers.size(); i++){
        stic(layers[i]->name.c_str());

        // Record the input for this layer (to be used in back propagation)
        result.layer_inputs.push_back(active_data);
        active_data = layers[i]->propagateForward(std::move(active_data));

        // std::cout << "After applying layer " << flow.names[i] << "\n";
        // for (uint layer = 0; layer < active_data.dim(2); layer++){
        //     printImage(active_data.slice(layer));
        // }
    }

    // Make sure the size of the last layer matches up with the label vector size
    assert(active_data.size() == output_labels.size());

    // Apply softmax to the output and store that as well
    global_timer.tic("Softmax");
    active_data = softMax(active_data);
    result.layer_inputs.push_back(active_data);
    global_timer.toc("Softmax");

    // Find the maximum probability
    auto max_idx = std::max_element(std::begin(active_data), std::end(active_data));

    // Construct the output data
    result.label_idx = std::distance(std::begin(active_data), max_idx);
    result.confidence = max(active_data)/sum(active_data);
    if (true_label != nullptr){
        result.loss = lossFcn(active_data, *true_label);
        result.true_label_idx = std::distance(std::begin(output_labels), std::find(std::begin(output_labels), std::end(output_labels), *true_label));
        // std::cout << "Label is " << +output_labels[result.label_idx] << " and the true label is " << +*true_label << ". Loss is " << result.loss << "\n";
    } 
    return result;
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::backwardsPropagation(const result_t& result, float learning_rate){
    STIC;
    // First we calculate the loss gradient with respect to the output values (assuming softmax and cross entropy)
    SimpleMatrix<float> dLdz = result.layer_inputs.back();
    dLdz[result.true_label_idx] -= 1;
    // for (size_t i = 0; i < flow.names.size(); i++){
    //     std::cout << "Stored input for layer " << i << ": " << flow.names.at(i) << "\n";
    //     const auto& M = result.layer_inputs[i];
    //     for (size_t layer = 0; layer < M.dim(2); layer++){
    //         printImage(M.slice(layer));
    //     }
    // }

    // For each of the layers, calculate the corresponding gradient and adjust the weights accordingly
    for (int i = layers.size()-1; i >= 0; i--){
        stic(layers[i]->name.c_str());
        const SimpleMatrix<float>& layer_input = result.layer_inputs[i];
        const SimpleMatrix<float>& layer_output = result.layer_inputs[i+1];
        dLdz = layers[i]->propagateBackward(layer_input, layer_output, dLdz, learning_rate, 0.01);
    }
}

} // namespace my_cnn