#pragma once

#include <stdexcept>
#include <type_traits>
#include <cnn_from_scratch/CUDA/CudaSafe.cuh>

namespace my_cnn{

template<typename T>
class CudaUniquePtr {
public:
    // Default constructor
    CudaUniquePtr() = default;

    // In place constructor
    template<typename... Args>
    CudaUniquePtr(Args... args) {
        T t_host(args...);
        CUDA_SAFE(cudaMalloc(&stored_, sizeof(T)));
        CUDA_SAFE(cudaMemcpy(stored_, &t_host, sizeof(T), cudaMemcpyHostToDevice));
    }

    // Move constructor
    CudaUniquePtr(CudaUniquePtr<T>&& other){
        stored_ = other.stored_;
        other.stored_ = nullptr;
    }

    // Move assignment operator
    CudaUniquePtr& operator=(const CudaUniquePtr&& other){
        if (other != *this){
            stored_ = other.stored_;
            other.stored_ = nullptr;
        }
        return *this;
    }

    // Destructor
    ~CudaUniquePtr(){
        cudaFree(stored_);
    }

    // Delete the copy and assignment operators
    CudaUniquePtr(const CudaUniquePtr<T>& other) = delete;
    CudaUniquePtr& operator=(const CudaUniquePtr<T>&) = delete;

    // Host dereference
    T getHostCopy() const {
        if (not stored_) throw std::runtime_error("Cannot dereference a CudaUniquePtr that was never initialized!");

        static_assert(std::is_trivially_constructible_v<T>);
        T host_T;
        CUDA_SAFE(cudaMemcpy(&host_T, stored_, sizeof(T), cudaMemcpyDeviceToHost));
        return host_T;
    }

    // Get access
    T* get() const{
        return stored_;
    }

private:
    T* stored_ = nullptr;
};
    
} // namespace my_cnn
