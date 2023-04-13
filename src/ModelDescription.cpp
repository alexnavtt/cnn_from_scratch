#include <assert.h>
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
    ConnectedLayer& layer = connected_layers.emplace_back();
    layer.biases.resize(output_size);
    flow.stages.push_back(ModelFlowMode::FULLY_CONNECTED);
    flow.names.push_back(name);
}

template<typename InputDataType, typename OutputDataType>
void ModelDescription<InputDataType, OutputDataType>::run(SimpleMatrix<InputDataType> input)
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
                kernels[idx].setInputData(active_data);
                *active_data = kernels[idx].convolve();
                break;

            case POOLING:
                *active_data = pooledMatrix(*active_data, pools[idx]);
                break;

            case FULLY_CONNECTED:
            {
                // If this is the first time at this layer, resize and apply random values
                ConnectedLayer& layer = connected_layers[idx];
                if (not layer.initialized){
                    layer.initialized = true;
                    layer.weights.resize(active_data->size(), layer.biases.size(), 1);
                    for (auto& v : layer.weights){
                        v = (float)rand() / (float)RAND_MAX;
                    }
                    for (auto& v : layer.biases){
                        v = (float)rand() / (float)RAND_MAX;
                    }
                }
                // Otherwise check to make sure the size is correct
                else{
                    assert(layer.weights.dims() == dim3(active_data->size(), layer.biases.size(), 1));
                }

                // Reshape the active data into a vector
                active_data->resize(1, active_data->size(), 1);

                // Matrix multiply to get the output values
                *active_data = active_data->matMul(layer.weights);

                // Add the biases
                *active_data += layer.biases;
                break;
            }

            case OUTPUT:
                break; /** TODO: */
        }

        std::cout << "After applying layer " << flow.names[i] << "\n";
        for (uint layer = 0; layer < active_data->dim(2); layer++){
            SimpleMatrix<float> image_layer(active_data->dims().slice(), (*active_data)[active_data->slice(layer)]);
            printImage(image_layer);
        }
    }
}

template class ModelDescription<float, std::string>;
template class ModelDescription<double, std::string>;
template class ModelDescription<char, std::string>;
template class ModelDescription<unsigned char, std::string>;
template class ModelDescription<int, std::string>;
template class ModelDescription<unsigned int, std::string>;

} // namespace my_cnn