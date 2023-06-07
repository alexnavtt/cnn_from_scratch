#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/Layers/Softmax.h"

extern cpp_timer::Timer global_timer;

namespace my_cnn{

template<typename InputDataType, typename OutputDataType>
Kernel& ModelDescription<InputDataType, OutputDataType>::addKernel
    (dim3 size, size_t count, ModelActivationFunction activation)
{
    auto& K = layers.emplace_back(new Kernel(size, count, 1));
    K->activation = activation;
    K->name = "Kernel_" + std::to_string(++kernel_count_);
    return *std::dynamic_pointer_cast<Kernel>(K);
}

template<typename InputDataType, typename OutputDataType>
Pooling& ModelDescription<InputDataType, OutputDataType>::addPooling
    (dim2 size, dim2 stride, PoolingType type)
{
    auto& P = layers.emplace_back(new Pooling(size, stride, type));
    P->name = "Pooling_" + std::to_string(++pooling_count_);
    return *std::dynamic_pointer_cast<Pooling>(P);
}

template<typename InputDataType, typename OutputDataType>
ConnectedLayer& ModelDescription<InputDataType, OutputDataType>::addConnectedLayer(size_t output_size, ModelActivationFunction activation)
{
    auto& C = layers.emplace_back(new ConnectedLayer(output_size));
    C->activation = activation;
    C->name = "FullyConnected_" + std::to_string(++fully_conn_count_);
    return *std::dynamic_pointer_cast<ConnectedLayer>(C);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::setOutputLabels(std::vector<OutputDataType> labels, OutputFunction output_function){
    output_labels = labels;
    output_function_ = output_function;

    switch (output_function_){
        case SOFTMAX:
            auto& S = layers.emplace_back(new Softmax(labels.size()));
            S->name = "SoftmaxOutput";
    }
}

template<typename InputDataType, typename OutputDataType>
float ModelDescription<InputDataType, OutputDataType>::lossFcn(const SimpleMatrix<double>& probabilities, size_t true_label_idx) const{
    constexpr float eps = 1e-5;
    
    switch (loss_function){
        default:
        case CROSS_ENTROPY:
            // Find the index of the true label
            return -1*log2(probabilities[true_label_idx] + eps);
    }
}

template<typename InputDataType, typename OutputDataType>
ModelResults<OutputDataType> ModelDescription<InputDataType, OutputDataType>::forwardPropagation(SimpleMatrix<InputDataType> input, OutputDataType* true_label)
{
    STIC;

    // If necessary, convert the input data type to double
    SimpleMatrix<double> active_data = std::move(input);

    // Create the output struct and store the input
    ModelResults<OutputDataType> result;
    result.layer_inputs.reserve(layers.size() + 1);
    result.layer_inputs.push_back(active_data);

    // For each stage of the model, apply the necessary step
    for (std::shared_ptr<ModelLayer>& layer : layers){
        stic(layer->name.c_str());

        // Record the input for this layer (to be used in back propagation)
        active_data = layer->propagateForward(std::move(active_data));
        result.layer_inputs.push_back(active_data);
    }

    // Make sure the size of the last layer matches up with the label vector size
    assert(active_data.size() == output_labels.size());

    result.label_idx  = active_data.maxIndex();
    result.confidence = active_data[result.label_idx];
    return result;
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::assignLoss(ModelResults<OutputDataType>& result, OutputDataType label){
    result.true_label_idx = std::distance(std::begin(output_labels), std::find(std::begin(output_labels), std::end(output_labels), label));
    result.true_label = label;
    result.knows_true_value = true;
    result.loss = lossFcn(result.layer_inputs.back(), result.true_label_idx);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::backwardsPropagation(ModelResults<OutputDataType>& result, OutputDataType label, float learning_rate){
    STIC;

    // Calculate the loss based on the given label
    assignLoss(result, label);

    // Provide the label to the output layer for gradient calculations
    assert(layers.back()->tag == OUTPUT);
    auto output_layer = std::dynamic_pointer_cast<Softmax>(layers.back());
    output_layer->assignTrueLabel(result.true_label_idx);

    // For each of the layers, calculate the corresponding gradient and adjust the weights accordingly
    SimpleMatrix<double> dLdz;
    for (int i = layers.size()-1; i >= 0; i--){
        stic(layers[i]->name.c_str());

        const SimpleMatrix<float>& layer_input  = result.layer_inputs.at(i);
        const SimpleMatrix<float>& layer_output = result.layer_inputs.at(i+1);
        dLdz = layers[i]->propagateBackward(layer_input, layer_output, dLdz, learning_rate, i == 0);
    }
    // std::cout << "\n============================================\n\n";
}

} // namespace my_cnn