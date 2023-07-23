#pragma once

#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
template <typename MatrixType>
void printImageBWImpl(const MatrixType& M, typename MatrixType::type max_val){
    using T = typename MatrixType::type;
    T min = my_cnn::min(M);
    T range = max_val - min;
    Dim3 idx;
    for (idx.x = 0; idx.x < M.dim().x; idx.x++){
        for (idx.y = 0; idx.y < M.dim().y; idx.y++){
            const T& val = M(idx);
            float dist = max_val - val;
            float mapped_val = max_val * (1 - dist/range);
            unsigned char code = std::min(static_cast<double>(mapped_val)/max_val * 255, 255.0);
            printf("\033[48;2;%d;%d;%dm  ", (int)code, (int)code, (int)code);
        }
        std::cout << "\033[0m\n";
    }
}

template <typename MatrixType>
void printImageColorImpl(const MatrixType& M, typename MatrixType::type max_val){
    using T = typename MatrixType::type;
    Dim3 idx;
    for (idx.x = 0; idx.x < M.dim().x; idx.x++){
        for (idx.y = 0; idx.y < M.dim().y; idx.y++){
            std::stringstream ss;
            ss << "\033[48;2";
            for (idx.z = 0; idx.z < 3; idx.z++){
                const T& val = M(idx);
                unsigned char code = std::min(static_cast<double>(val)/max_val * 255, 255.0);
                ss << ";" << +code;
            }
            ss << "m   ";
            std::cout << ss.str();
        }
        std::cout << "\033[0m\n";
    }
}

template<typename MatrixType>
void printImage(const MatrixType& M){
    using T = typename MatrixType::type;
    typename std::remove_const_t<T> max = (std::is_integral_v<T> ? std::numeric_limits<unsigned char>::max() : my_cnn::max(M));
    if (max < 1) max = 1;

    if (M.dim().z == 3)
        printImageColorImpl<MatrixType>(M, max); 
    else
        printImageBWImpl<MatrixType>(M, max);
}

} // namespace my_cnn
