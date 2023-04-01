#pragma once

#include "cnn_from_scratch/SimpleMatrix.h"

namespace my_cnn{
    
template <typename T>
void printImageBWImpl(const SimpleMatrix<T>& M, T max_val){
    for (size_t row = 0; row < M.dims().x; row++){
        for (size_t col = 0; col < M.dims().y; col++){
            const T& val = M(row, col, 0);
            unsigned char code = std::min(static_cast<float>(val)/max_val * 255, 255.0f);
            printf("\033[48;2;%d;%d;%dm  ", (int)code, (int)code, (int)code);
        }
        std::cout << "\033[0m\n";
    }
}

template <typename T>
void printImageColorImpl(const SimpleMatrix<T>& M, T max_val){
    for (size_t row = 0; row < M.dims().x; row++){
        for (size_t col = 0; col < M.dims().y; col++){
            std::stringstream ss;
            ss << "\033[48;2";
            for (size_t channel = 0; channel < 3; channel++){
                const T& val = M(row, col, channel);
                unsigned char code = std::min(static_cast<float>(val)/max_val * 255, 255.0f);
                ss << ";" << +code;
            }
            ss << "m   ";
            std::cout << ss.str();
        }
        std::cout << "\033[0m\n";
    }
}

template<typename T>
void printImage(const SimpleMatrix<T>& M){
    if constexpr (std::is_integral_v<T>){
        if (M.dim(2) == 3)
            printImageColorImpl<T>(M, std::numeric_limits<unsigned char>::max()); 
        else
            printImageBWImpl<T>(M, std::numeric_limits<unsigned char>::max());
    }else if (std::is_floating_point_v<T>){
        if (M.dim(2) == 3)
            printImageColorImpl<T>(M, 1.0); 
        else
            printImageBWImpl<T>(M, 1.0);
    }
}

} // namespace my_cnn
