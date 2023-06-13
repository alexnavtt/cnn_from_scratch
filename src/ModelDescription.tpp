#include <fstream>
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/timerConfig.h"
#include "cnn_from_scratch/Serialization.h"
#include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/Layers/Softmax.h"

extern cpp_timer::Timer global_timer;

namespace my_cnn{

template<typename InputDataType, typename OutputDataType>
Kernel& ModelDescription<InputDataType, OutputDataType>::addKernel
    (dim3 size, size_t count)
{
    auto& K = layers.emplace_back(new Kernel(size, count, 1));
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
ConnectedLayer& ModelDescription<InputDataType, OutputDataType>::addConnectedLayer(size_t output_size)
{
    auto& C = layers.emplace_back(new ConnectedLayer(output_size));
    C->name = "FullyConnected_" + std::to_string(++fully_conn_count_);
    return *std::dynamic_pointer_cast<ConnectedLayer>(C);
}

template<typename InputDataType, typename OutputDataType>
Activation& ModelDescription<InputDataType, OutputDataType>::addActivation(ModelActivationFunction activation)
{
    auto& A = layers.emplace_back(new Activation(activation));
    A->name = "Activation_" + std::to_string(++activation_count_);
    return *std::dynamic_pointer_cast<Activation>(A);
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
bool ModelDescription<InputDataType, OutputDataType>::saveModel(std::string filename) {
    STIC;
    std::ofstream f(filename);

    int i = 0;
    try{
        for (auto& layer : layers){
            f << "(" << i++ << ") ----------\n";
            f << layer->serialize();
        }
    }catch(...){
        std::cout << "Something went wrong, file was not saved correctly\n";
        return false;
    }

    f << "\nOutput labels " << output_labels.size() << "\n";
    for (auto& label : output_labels){
        f << std::to_string(label) << "\n";
    }
    f << std::endl;

    return true;
}

template<typename InputDataType, typename OutputDataType>
bool ModelDescription<InputDataType, OutputDataType>::loadModel(std::string filename) {
    STIC;
    std::ifstream f(filename);

    std::string line;
    while (std::getline(f, line)){
        // Read the next line without extracting it
        std::string label = serialization::readLine(f);
        
        // Determine the type of layer coming next
        try{
            if (label == "Kernel"){
                Kernel& K = addKernel({0, 0, 0}, 1, LINEAR);
                if (not K.deserialize(f)) return false;
            }

            else if (label == "Pooling"){
                Pooling& P = addPooling({0, 0}, {0, 0}, MAX);
                if (not P.deserialize(f)) return false;
            }

            else if (label == "Connected Layer"){
                ConnectedLayer& C = addConnectedLayer(0);
                if (not C.deserialize(f)) return false;
            }
            
            else if (label == "Softmax"){
                // Determine the number of output labels
                serialization::clearLine(f);
                serialization::clearLine(f);
                int size = serialization::expect<int>(f, "Output labels");

                // Load the output vector
                std::vector<OutputDataType> labels(size);
                for (int i = 0; i < size; i++){
                    OutputDataType label;
                    f >> label;
                    serialization::clearLine(f);
                    labels[i] = label;
                }

                // Set the vector with softmax activation
                setOutputLabels(labels, SOFTMAX);
            }
        }
        
        catch(std::runtime_error& e){
            std::cout << "loadModel runtime error: " << e.what() << "\n";
            return false;
        }
    }

    return true;
}

template<typename InputDataType, typename OutputDataType>
float ModelDescription<InputDataType, OutputDataType>::lossFcn(const SimpleMatrix<double>& probabilities, size_t true_label_idx) const{
    constexpr float eps = 1e-5;
    
    switch (loss_function){
        default:
        case CROSS_ENTROPY:
            // Find the index of the true label
            double loss = -1*log2(probabilities[true_label_idx] + eps);
            return std::max(loss, 0.0);
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
    result.loss = lossFcn(result.layer_inputs.back(), result.true_label_idx);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::backwardsPropagation(ModelResults<OutputDataType>& result, OutputDataType label, float learning_rate){
    STIC;

    // Calculate the loss based on the given label
    assignLoss(result, label);

    // Provide the label to the output layer for gradient calculations
    assert(layers.back()->getType() == OUTPUT);
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
}

} // namespace my_cnn