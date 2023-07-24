#include <iostream>
#include <cuda_runtime.h>

#include "cnn_from_scratch/Matrix/dim.h"
#include "cnn_from_scratch/CUDA/Matrix/convolve.cuh"

__global__ void kernel_test(my_cnn::Dim3 dim){
    printf("The value is (%d, %d, %d)\n", dim.x, dim.y, dim.z);
}

int main(){

    my_cnn::testConvolveReduce();

    std::cout << "=========\n";

    my_cnn::testConvolveDot();

    return 0;
}