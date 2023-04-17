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
    connected_layers.emplace_back(output_size);
    flow.stages.push_back(ModelFlowMode::FULLY_CONNECTED);
    flow.names.push_back(name);
}

template<typename InputDataType, typename OutputDataType>
float ModelDescription<InputDataType, OutputDataType>::lossFcn(const SimpleMatrix<float>& probabilities, const OutputDataType& true_label) const{
    switch (loss_function){
        default:
        case CROSS_ENTROPY:
            // Find the index of the true label
            size_t true_idx = std::distance(std::begin(output_labels), std::find(std::begin(output_labels), std::end(output_labels), true_label));
            return -1*log2(probabilities[true_idx]);
    }
}

template<typename InputDataType, typename OutputDataType>
ModelResults<OutputDataType> ModelDescription<InputDataType, OutputDataType>::forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label)
{
    // If necessary, convert the input data type to float
    SimpleMatrix<float> active_data = std::move(input);

    std::cout << "Input image is \n";
    for (uint layer = 0; layer < active_data.dim(2); layer++){
        printImage(active_data.slice(layer));
    }

    // Create the output struct and store the input
    ModelResults<OutputDataType> result;
    result.layer_inputs.reserve(flow.indices.size() + 1);

    // For each stage of the model, apply the necessary step
    for (size_t i = 0; i < flow.stages.size(); i++){
        ModelFlowMode stage = flow.stages[i];
        size_t idx = flow.indices[i];

        // Record the input for this layer (to be used in back propagation)

        switch (stage){
            case KERNEL:
                result.layer_inputs.push_back(active_data);
                active_data = kernels[idx].propagateForward(active_data);
                break;

            case POOLING:
                result.layer_inputs.push_back(active_data);
                active_data = pools[idx].propagateForward(active_data);
                break;

            case FULLY_CONNECTED:
            {
                // Reshape to a column vector before passing it to the fully connected layer
                active_data.reshape(active_data.size(), 1, 1);
                result.layer_inputs.push_back(active_data);
                active_data = connected_layers[idx].propagateForward(active_data);
                break;
            }
        }

        std::cout << "After applying layer " << flow.names[i] << "\n";
        for (uint layer = 0; layer < active_data.dim(2); layer++){
            printImage(active_data.slice(layer));
        }
    }

    // Make sure the size of the last layer matches up with the label vector size
    assert(active_data.size() == output_labels.size());

    // Apply softmax to the output and store that as well
    active_data = exp(active_data - max(active_data));
    active_data /= sum(active_data);
    result.layer_inputs.push_back(active_data);

    // Find the maximum probability
    auto max_idx = std::max_element(std::begin(active_data), std::end(active_data));

    // Construct the output data
    result.label_idx = std::distance(std::begin(active_data), max_idx);
    result.label = output_labels[result.label_idx];
    result.loss = true_label ? lossFcn(active_data, *true_label) : -1.0f;
    return result;
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::backwardsPropagation(const result_t& result, float learning_rate){
    // First we calculate the loss gradient with respect to the output values (assuming softmax and cross entropy)
    SimpleMatrix<float> dLdz = result.layer_inputs.back();
    dLdz[result.label_idx] -= 1;

    int ii = 0;
    for (const SimpleMatrix<float>& M : result.layer_inputs){
        std::cout << "Stored input for layer " << ii << ": " << flow.names[ii++] << "\n";
        for (size_t layer = 0; layer < M.dim(2); layer++){
            printImage(M.slice(layer));
        }
    }

    // For each of the layers, calculate the corresponding gradient and adjust the weights accordingly
    for (int i = result.layer_inputs.size()-2; i >= 0; i--){
        const SimpleMatrix<float>& layer_input = result.layer_inputs[i];
        size_t idx = flow.indices[i];

        switch (flow.stages[i]){
            case FULLY_CONNECTED:
                dLdz = connected_layers[idx].propagateBackward(layer_input, dLdz, learning_rate);
                break;

            case KERNEL:
                dLdz.reshape(kernels[idx].outputSize(layer_input));
                break;

            case POOLING:
                dLdz = pools[idx].propagateBackward(layer_input, dLdz, learning_rate);
                break;
        }
    }
}

} // namespace my_cnn