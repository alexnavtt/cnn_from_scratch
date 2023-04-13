#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"

namespace my_cnn{
    
template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addKernel
    (Kernel kernel, std::string_view name)
{
    flow.indices.push_back(kernels.size());
    kernels.push_back(kernel);
    flow.stages.push_back(ModelFlowMode::KERNEL);
    flow.names.push_back(name);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addPooling
    (Pooling pool, std::string_view name)
{
    flow.indices.push_back(pools.size());
    pools.push_back(pool);
    flow.stages.push_back(ModelFlowMode::POOLING);
    flow.names.push_back(name);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::addConnectedLayer
    (size_t output_size, std::string_view name)
{
    flow.indices.push_back(connected_layers.size());
    ConnectedLayer& layer = connected_layers.emplace_back(output_size);
    flow.stages.push_back(ModelFlowMode::FULLY_CONNECTED);
    flow.names.push_back(name);
}

template<typename InputDataType, typename OutputDataType>
float ModelDescription<InputDataType, OutputDataType>::lossFcn(const std::valarray<float>& probabilities, const OutputDataType& true_label) const{
    switch (loss_function){
        default:
        case CROSS_ENTROPY:
            // Find the index of the true label
            size_t true_idx = std::distance(std::begin(output_labels), std::find(std::begin(output_labels), std::end(output_labels), true_label));
            return -1*log10(probabilities[true_idx]);
    }
}

template<typename InputDataType, typename OutputDataType>
ModelResults<OutputDataType> ModelDescription<InputDataType, OutputDataType>::forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label)
{
    SimpleMatrix<float> kernel_copy;
    SimpleMatrix<float>* active_data = nullptr;
    if constexpr (not std::is_same_v<InputDataType, float>){
        kernel_copy = input;
        active_data = &kernel_copy;
    }else{
        active_data = &input;
    }
    std::cout << "Input image is \n";
    for (uint layer = 0; layer < active_data->dim(2); layer++){
        SimpleMatrix<float> image_layer(active_data->dims().slice(), (*active_data)[active_data->slice(layer)]);
        printImage(image_layer);
    }

    // For each stage of the model, apply the necessary step
    for (size_t i = 0; i < flow.stages.size(); i++){
        ModelFlowMode stage = flow.stages[i];
        size_t idx = flow.indices[i];

        switch (stage){
            case KERNEL:
                *active_data = kernels[idx].apply(*active_data);
                break;

            case POOLING:
                *active_data = pooledMatrix(*active_data, pools[idx]);
                break;

            case FULLY_CONNECTED:
            {
                // Reshape to a column vector before passing it to the fully connected layer
                active_data->resize(active_data->size(), 1, 1);
                *active_data = connected_layers[idx].apply(*active_data);
                break;
            }
        }

        std::cout << "After applying layer " << flow.names[i] << "\n";
        for (uint layer = 0; layer < active_data->dim(2); layer++){
            SimpleMatrix<float> image_layer(active_data->dims().slice(), (*active_data)[active_data->slice(layer)]);
            printImage(image_layer);
        }
    }

    // Make sure the size of the last layer matches up with the label vector size
    assert(active_data->size() == output_labels.size());

    // Create the output struct
    ModelResults<OutputDataType> result;

    // Apply softmax to the output and store that as well
    *active_data = std::exp(*active_data - active_data->max());
    *active_data /= active_data->sum();
    result.softmax_output = std::move(*active_data);

    // Find the maximum probability
    float* max_idx = std::max_element(std::begin(result.softmax_output), std::end(result.softmax_output));

    // Construct the output data
    result.label_idx = std::distance(std::begin(result.softmax_output), max_idx);
    result.label = output_labels[result.label_idx];
    result.loss = true_label ? lossFcn(result.softmax_output, *true_label) : -1.0f;
    return result;
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::backwardsPropagation(const result_t& result, float learning_rate){
    // First we calculate the loss gradient with respect to the output values
    std::valarray<float> dLdz = result.softmax_output;
    dLdz[result.label_idx] -= 1;

    // For each of the layers, calculate the corresponding gradient and adjust the weights accordingly
    for (size_t i = flow.indices.size()-1; i >= 0; i--){
        switch (flow.stages[i]){
            case FULLY_CONNECTED:
                ConnectedLayer& layer = connected_layers[i];
                // dWd
                break;
        }
    }
}

} // namespace my_cnn