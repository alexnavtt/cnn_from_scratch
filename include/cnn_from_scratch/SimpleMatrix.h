#pragma once

#include <math.h>
#include <utility>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <type_traits>

#include "cnn_from_scratch/dim3.h"
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/SubMatrixView.h"

namespace my_cnn{

#define THROW_SIZE_EXCEPTION { \
    std::stringstream ss;      \
    ss << "Matrix size mismatch. Sizes are (" \
       << dim_.x << ", " << dim_.y << ", " << dim_.z \
       << ") and (" \
       << other.dim_.x << ", " << other.dim_.y << ", " << other.dim_.z \
       << ")\n";    \
       throw MatrixSizeException(ss.str()); \
}

#define ADD_MATRIX_CONST_OPERATOR(op) \
    template<typename Other> \
    SimpleMatrix<T> operator op(const Other& other) const{ \
        using namespace std::literals; \
        if (not sizeCheck(other)) \
            throw MatrixSizeException("my_cnn::SimpleMatrix: Failure in operator \""s + #op + "\", size mismatch"s); \
        SimpleMatrix<T> out; \
        auto out_varr = static_cast<std::valarray<T>*>(&out); \
        auto curr_varr = static_cast<const std::valarray<T>*>(this);  \
        *out_varr = *curr_varr op other; \
        out.dim_ = dim_; \
        return out; \
    } \
    \
    template<typename Other> \
    friend SimpleMatrix<T> operator op(const Other& other, const SimpleMatrix<T>& M){ \
        return M.operator op(other); \
    }

#define ADD_MATRIX_MODIFYING_OPERATOR(op) \
    template<typename Other> \
    SimpleMatrix<T>& operator op(const Other& other){ \
        using namespace std::literals; \
        if (not sizeCheck(other)) \
            throw MatrixSizeException("my_cnn::SimpleMatrix: Failure in operator \""s + #op + "\", size mismatch"s); \
        static_cast<std::valarray<T>&>(*this) op other; \
        return *this; \
    }

template<typename T>  
class SimpleMatrix : public std::valarray<T>{
    // Make all temlplates of SimpleMatrix friends
    template <typename Other>
    friend class SimpleMatrix;
    
public:

    // Constructors
    SimpleMatrix(){};

    SimpleMatrix(dim3 dim, T initial_val=T{}):
    std::valarray<T>(initial_val, dim.x*dim.y*dim.z),
    dim_(dim)
    {}

    template<typename Other>
    bool sizeCheck(const SimpleMatrix<Other>& other) const noexcept{
        return other.dim_ == dim_;
    }

    template<typename Other>
    bool sizeCheck(const std::valarray<Other>& v) const noexcept{
        return v.size() == this->size();
    }

    template<typename Other>
    bool sizeCheck(const Other& v) const noexcept{
        return true;
    }

    SimpleMatrix<T> subMatCopy(dim3 idx, dim3 sub_dim) const;

    std::gslice subMatIdx(dim3 start, dim3 size) const{
        return std::gslice(
            getIndex(start),
            {size.z, size.y, size.x},
            {dim_.y*dim_.x, dim_.x, 1}
        );
    }

    // Type conversion
    template<typename Other>
    SimpleMatrix<T>& operator=(const SimpleMatrix<Other>& M);

    // Value setting
    SimpleMatrix<T>& operator=(std::valarray<T>&& v){
        if (v.size() == this->size())
            static_cast<std::valarray<T>&>(*this) = std::forward<std::valarray<T>>(v);
        else
            throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
        return *this;
    }

    void setEntries(std::valarray<T>&& v){
        if (v.size() == this->size())
            static_cast<std::valarray<T>&>(*this) = v[
                std::gslice(
                    0, 
                    {dim_.z, dim_.y, dim_.x},
                    {dim_.x*dim_.y, 1, dim_.y}
                )
            ];
        else
            throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
    }

    ADD_MATRIX_CONST_OPERATOR(+);
    ADD_MATRIX_CONST_OPERATOR(-);
    ADD_MATRIX_CONST_OPERATOR(*);
    ADD_MATRIX_CONST_OPERATOR(/);

    ADD_MATRIX_MODIFYING_OPERATOR(+=);
    ADD_MATRIX_MODIFYING_OPERATOR(-=);
    ADD_MATRIX_MODIFYING_OPERATOR(*=);
    ADD_MATRIX_MODIFYING_OPERATOR(/=);

    template<typename T2>
    friend std::ostream& operator<<(std::ostream& os, const SimpleMatrix<T2>& M);

    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    size_t getIndex(dim3 dim) const;

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0);
    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    T& operator()(dim3 idx);
    const T& operator()(dim3 idx) const;

    uint dim(size_t idx) const{
        return dim_.data[idx];
    }

    const dim3& dims() const{
        return dim_;
    }

    SimpleMatrix<T> abs() const {
        if constexpr (std::is_unsigned_v<T>)
            return *this;
        else{
            SimpleMatrix<T> out(dim_);
            return out = std::abs(*this);
        }
    }

    // SimpleMatrix<T> slice(unsigned idx) const{
    //     return subMat({0, 0, idx}, {dim_.x, dim_.y, 1});
    // }

    // std::valarray<T> channelSum() const{
    //     std::valarray<T> output(dim_.z);
    //     for (unsigned z = 0; z < dim_.z; z++){
    //         output[z] = slice(z).sum();
    //     }
    //     return output;
    // }

protected:
    dim3 dim_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const my_cnn::SimpleMatrix<T>& M){
    int default_precision = std::cout.precision();
    int max_width;
    if constexpr (std::is_integral_v<T>){
        const T max = abs(M).max();
        max_width = ceil(log10(max));
    }else{
        std::cout << std::setprecision(3) << std::fixed;
        max_width = 4;
    }

    for (int i = 0; i < M.dim_.x; i++){
        for (int k = 0; k < M.dim_.z; k++){
            os << "[";
            for (int j = 0; j < M.dim_.y; j++){
                const auto& val = M(i, j, k);
                os << (val < 0 ? "" : " ") << std::setw(max_width) << +val << (j == M.dim_.y - 1 ? "]" : ",");
            }
            os << "   ";
        }
        os << "\n";
    }

    std::cout << std::setprecision(default_precision);
    std::cout << std::defaultfloat;

    return os;
}

} // namespace my_cnn
