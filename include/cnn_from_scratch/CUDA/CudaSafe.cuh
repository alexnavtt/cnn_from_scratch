#pragma once

#include <cuda_runtime.h>

#define CUDA_SAFE_NAMED(command, name)                                  \
{cudaError_t err = command;                                             \
if (err != cudaSuccess){                                                \
    const char* err_s = cudaGetErrorString(err);                        \
    printf("\n\tCuda Error [%d]\n\tFailed Command: \"%s\"\n\tError: %s\n\tAt line %d of %s\n", (int)(err), name, err_s, __LINE__, __FILE__); \
    throw std::runtime_error("CUDA runtime exception");                 \
}}\

#define CUDA_SAFE(command) CUDA_SAFE_NAMED(command, #command)
