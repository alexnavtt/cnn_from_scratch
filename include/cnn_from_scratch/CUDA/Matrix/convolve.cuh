#include <vector>
#include <algorithm>
#include "cnn_from_scratch/CUDA/CudaSafe.cuh"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn {

struct BatchDescription{
    uint batch_size;
    uint num_input_channels;
    uint num_output_channels;
    uint output_x;
    uint output_y;
    uint input_x;
    uint input_y;
    uint kernel_x;
    uint kernel_y;
};

// For each input pixel, calculate the product between it and the correspoding kernel pixel
__global__ void convolve_dot_kernel(BatchDescription params, double* input_data, double* kernel_data, double* output){
    
    // Figure out indexing for this thread
    const unsigned int output_pixel_idx   = (threadIdx.x + blockDim.x * blockIdx.x) % (params.output_x * params.output_y); 
    const unsigned int batch_idx          = (threadIdx.x + blockDim.x * blockIdx.x) / (params.output_x * params.output_y);
    const unsigned int kernel_pixel_idx   = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int input_channel_idx  = threadIdx.z % params.num_input_channels;
    const unsigned int output_channel_idx = threadIdx.z / params.num_input_channels;

    // Get the 2D x-y coordinate of the kernel pixel
    const unsigned int kernel_x_idx = kernel_pixel_idx % params.kernel_x;
    const unsigned int kernel_y_idx = kernel_pixel_idx / params.kernel_x;

    // Get the 2D x-y coordinate of the output pixel
    const unsigned int output_x_idx = output_pixel_idx % params.output_x;
    const unsigned int output_y_idx = output_pixel_idx / params.output_x;

    if (output_x_idx >= params.output_x || output_y_idx >= params.output_y 
     || kernel_x_idx >= params.kernel_x || kernel_y_idx >= params.kernel_y
     || batch_idx >= params.batch_size) return;

    // Use those values to get the 2D x-y coordinate of the input pixel
    const unsigned int input_x_idx = output_x_idx + kernel_x_idx;
    const unsigned int input_y_idx = output_y_idx + kernel_y_idx;

    // Get the offset to each of the relevant input and kernel data points
    size_t input_data_offset = batch_idx * params.num_input_channels * params.input_x * params.input_y 
                             + input_channel_idx * params.input_x * params.input_y 
                             + input_y_idx * params.input_x
                             + input_x_idx;

    size_t kernel_data_offset = output_channel_idx * params.num_input_channels * params.kernel_x * params.kernel_y
                              + input_channel_idx * params.kernel_x * params.kernel_y
                              + kernel_pixel_idx;

    size_t output_data_offset = batch_idx * params.num_output_channels * params.output_x * params.output_y * params.num_input_channels * params.kernel_x * params.kernel_y
                              + output_channel_idx * params.output_x * params.output_y * params.num_input_channels * params.kernel_x * params.kernel_y
                              + output_y_idx * params.output_x * params.num_input_channels * params.kernel_x * params.kernel_y
                              + output_x_idx * params.num_input_channels * params.kernel_x * params.kernel_y
                              + input_channel_idx * params.kernel_x * params.kernel_y
                              + kernel_y_idx * params.kernel_x 
                              + kernel_x_idx;

    printf("Thread Index: (%u, %u, %u) | Block Index: (%u, %u, %u) | Output Pixel: (%u, %u) | Kernel Pixel: (%u, %u) : %.1f| Output Data Offset: %2d | Input Data Offset: %2d | Kernel Data Offset: %2d\n", 
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        output_x_idx, output_y_idx,
        kernel_x_idx, kernel_y_idx, kernel_data[kernel_data_offset],
        (int)output_data_offset, (int)input_data_offset, (int)kernel_data_offset);

    output[output_data_offset] = input_data[input_data_offset] * kernel_data[kernel_data_offset];
}

void testConvolveDot(){

    BatchDescription params;
    params.batch_size = 1;

    SimpleMatrix<double> input_data({3, 3, 2}, 
        {1, 2, 3,
        -5, 9, 4,
         3, 7, 5,
         
         3, 5,-1,
         2, 2, 7,
         0, 5, 1});

    double* device_input_data;
    CUDA_SAFE(cudaMalloc(&device_input_data, sizeof(double) * input_data.size()));
    CUDA_SAFE(cudaMemcpy(device_input_data, &input_data[0], input_data.size() * sizeof(double), cudaMemcpyHostToDevice));
    params.input_x = input_data.dim().x;
    params.input_y = input_data.dim().y;
    params.num_input_channels = input_data.dim().z;

    SimpleMatrix<double> kernel({2, 2, 2},
        {-1, 1,
          2, 0, 
          
          3, 2,
          0, 1});

    double* device_kernel_data;
    CUDA_SAFE(cudaMalloc(&device_kernel_data, sizeof(double) * kernel.size()));
    CUDA_SAFE(cudaMemcpy(device_kernel_data, &kernel[0], kernel.size() * sizeof(double), cudaMemcpyHostToDevice));
    params.kernel_x = kernel.dim().x;
    params.kernel_y = kernel.dim().y;
    params.num_output_channels = kernel.dim().z / params.num_input_channels;

    params.output_x = params.input_x - params.kernel_x + 1;
    params.output_y = params.input_y - params.kernel_y + 1;

    double* device_output_data;
    size_t output_data_size = sizeof(double) * params.output_x * params.output_y * params.num_output_channels * params.batch_size * kernel.size();
    CUDA_SAFE(cudaMalloc(&device_output_data, output_data_size));

    uint total_x_threads = params.output_x * params.output_y * params.batch_size;
    uint total_y_threads = kernel.dim().x * kernel.dim().y;
    uint total_z_threads = kernel.dim().z;

    dim3 block_size;
    block_size.z = total_z_threads;
    block_size.y = std::min((int) (1024 / block_size.z), kernel.dim().x * kernel.dim().y);
    block_size.x = std::clamp((int) (1024 / (block_size.y * block_size.z)), 1, (int)(params.batch_size * params.output_x * params.output_y));

    dim3 grid_size;
    grid_size.x = total_x_threads / block_size.x + 1;
    grid_size.y = total_y_threads / block_size.y + 1;
    grid_size.z = 1;

    std::cout << "Block size is (" << block_size.x << ", " << block_size.y << ", " << block_size.z << ")\n";
    std::cout << "Grid size is (" << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << ")\n";

    convolve_dot_kernel<<<grid_size, block_size>>>(params, device_input_data, device_kernel_data, device_output_data);
    cudaDeviceSynchronize();
    CUDA_SAFE_NAMED(cudaGetLastError(), "convolve_dot_kernel");

    double output_data[output_data_size/sizeof(double)];
    CUDA_SAFE(cudaMemcpy(&output_data, device_output_data, sizeof(output_data), cudaMemcpyDeviceToHost));

    std::cout << "Input: " << input_data;
    std::cout << "Kernel: " << kernel;
    size_t out_idx = 0;
    for (int y = 0; y < params.output_y; y++){
        for (int x = 0; x < params.output_x; x++){
            printf("Output pixel (%d, %d):\n", x, y);
            for (int i = 0; i < params.kernel_x * params.kernel_y * params.num_input_channels; i++){
                std::cout << "|" << output_data[out_idx++] << "|\n";
            }
        }
    }

    cudaFree(device_input_data);
    cudaFree(device_kernel_data);
    cudaFree(device_output_data);
}

__global__ void convolve_reduction_kernel(double* convolved_data, size_t data_block_num_elements, double* output, size_t num_data_blocks) {
    
    extern __shared__ double summation_block[];

    // Determine which data block (i.e. output pixel) we're working with
    const unsigned int data_block_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (data_block_idx >= num_data_blocks) return;

    // Calculate the data offset
    const unsigned int block_data_idx  = threadIdx.y;   // index within each data block
    const unsigned int local_block_idx = threadIdx.x;   // which data block within this CUDA block 

    const size_t data_offset    = data_block_idx  * data_block_num_elements + block_data_idx;
    const size_t local_data_idx = local_block_idx * data_block_num_elements + block_data_idx; 

    // Extract the local data block corresponding to a single output pixel
    summation_block[local_data_idx] = convolved_data[data_offset];
    __syncthreads();

    // I'm assuming that the optimizer will convert this to all the bit shift stuff
    int largest_power_of_2 = powf(2, (int)log2f(data_block_num_elements));

    // Deal with all values that don't fit cleanly into a power of 2
    if (block_data_idx + largest_power_of_2 < data_block_num_elements){
        summation_block[local_data_idx] += summation_block[local_data_idx + largest_power_of_2];
    }
    largest_power_of_2 /= 2;

    // Reduction algorithm
    while (largest_power_of_2 > block_data_idx){
        summation_block[local_data_idx] += summation_block[local_data_idx + largest_power_of_2];
        largest_power_of_2 /= 2;
        __syncthreads();
    }

    if (block_data_idx) return;

    output[data_block_idx] = summation_block[local_block_idx * data_block_num_elements];
}

void testConvolveReduce(){
    double test_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t data_block_num_elements = 11;
    size_t num_data_blocks = 3;

    double* device_output;
    CUDA_SAFE(cudaMalloc(&device_output, sizeof(double)*num_data_blocks));

    double* device_data;
    CUDA_SAFE(cudaMalloc(&device_data, sizeof(test_data)));
    CUDA_SAFE(cudaMemcpy(device_data, &test_data, sizeof(test_data), cudaMemcpyHostToDevice));

    dim3 block_size;
    // block_size.x = std::min(num_data_blocks, 1024 / data_block_num_elements);
    block_size.x = 1;
    block_size.y = data_block_num_elements;
    block_size.z = 1;

    dim3 grid_size;
    grid_size.x = num_data_blocks / block_size.x + 1;

    std::cout << "Block size is (" << block_size.x << ", " << block_size.y << ", " << block_size.z << ")\n";
    std::cout << "Grid size is (" << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << ")\n";

    size_t shared_memory_size = data_block_num_elements * num_data_blocks * sizeof(double);
    convolve_reduction_kernel<<<grid_size, block_size, shared_memory_size>>>(device_data, data_block_num_elements, device_output, num_data_blocks);
    cudaDeviceSynchronize();
    CUDA_SAFE_NAMED(cudaGetLastError(), "convolve_reduction_kernel"); 

    double result[num_data_blocks];
    CUDA_SAFE(cudaMemcpy(&result, device_output, sizeof(double)*num_data_blocks, cudaMemcpyDeviceToHost));

    std::cout << "The results are\n";
    for (int i = 0; i < num_data_blocks; i++){
        printf("\t%.f\n", result[i]);
    }

    cudaFree(device_output);
    cudaFree(device_data);
}

void convolve(const std::vector<SimpleMatrix<double>>& inputs, const SimpleMatrix<double>& kernel) {
    // Trivial case
    if (inputs.empty()) return;

    // Get the matrix dimension
    const Dim3 dim = inputs[0].dim();

    // Calculate the output dimensions
    const Dim3 output_dim(dim.x - kernel.dim().x + 1, dim.y - kernel.dim().y + 1, dim.z);

    // Allocate space for the inputs on the device
    double* device_input_data;
    CUDA_SAFE(cudaMalloc(&device_input_data, sizeof(double) * inputs.size() * inputs[0].size()));

    // Copy the data to the device
    for (const SimpleMatrix<double>& mat : inputs){
        CUDA_SAFE(cudaMemcpy(device_input_data, &mat[0], sizeof(double) * mat.size(), cudaMemcpyHostToDevice));
    }

    // Allocate space for the kernel on the device and copy
    double* device_kernel_data;
    CUDA_SAFE(cudaMalloc(&device_kernel_data, sizeof(double) * kernel.size()));
    CUDA_SAFE(cudaMemcpy(device_kernel_data, &kernel[0], sizeof(double) * kernel.size(), cudaMemcpyHostToDevice));

    // Allocate space for the result of the calculation on the device
    double* dot_product_output;
    CUDA_SAFE(cudaMalloc(&dot_product_output, sizeof(double) * kernel.dim().x * kernel.dim().y * output_dim.size()));

    // Run the kernel
    // convolve_kernel<<<1, 1>>>(device_input_data, dim, device_kernel_data, kernel.dim(), 1, 1);
    cudaDeviceSynchronize();
    CUDA_SAFE_NAMED(cudaGetLastError(), "convolve_kernel");
}
    
} // namespace my_cnn
