#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"

namespace my_cnn{
    
template<typename InputDataType>
void ModelDescription<InputDataType>::addKernel
    (Kernel kernel, std::string_view name)
{
    flow.indices.push_back(kernels.size());
    kernels.push_back(kernel);
    flow.stages.push_back(ModelFlowMode::KERNEL);
    flow.names.push_back(name);
}

template<typename InputDataType>
void ModelDescription<InputDataType>::addPooling
    (Pooling pool, std::string_view name)
{
    flow.indices.push_back(pools.size());
    pools.push_back(pool);
    flow.stages.push_back(ModelFlowMode::POOLING);
    flow.names.push_back(name);
}

template<typename InputDataType>
void ModelDescription<InputDataType>::run(SimpleMatrix<InputDataType> input)
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
                break; /** TODO: */

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

template class ModelDescription<float>;
template class ModelDescription<double>;
template class ModelDescription<char>;
template class ModelDescription<unsigned char>;
template class ModelDescription<int>;
template class ModelDescription<unsigned int>;

} // namespace my_cnn