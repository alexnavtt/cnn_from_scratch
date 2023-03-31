#pragma once

#include <math.h>
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

#define ADD_VALARRAY_OPERATOR(op)                                       \
    template<typename Other>                                            \
    SimpleMatrix<T> operator op (const Other& other){                   \
        SimpleMatrix<T> new_mat = *this;                                \
        if constexpr (std::is_same_v<Other, SimpleMatrix<T>>){          \
            if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;               \
            new_mat.data_ op##= other.data_;                            \
        }else{                                                          \
            new_mat.data_ op##= other;                                  \
        }                                                               \
        return new_mat;                                                 \
    }                                                                   \
    template<typename Other>                                            \
    SimpleMatrix<T> operator op##= (const Other& other){                \
        if constexpr (std::is_same_v<Other, SimpleMatrix<T>>){          \
            if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;               \
            data_ op##= other.data_;                                    \
        }else{                                                          \
            data_ op##= other;                                          \
        }                                                               \
        return *this;                                                   \
    }

template<typename T>  
class SimpleMatrix{
public:
    SimpleMatrix(dim3 dim):
    dim_(dim),
    data_((T)0, dim.x*dim.y*dim.z)
    {}

    SimpleMatrix<T> subMatrix(dim3 idx, dim3 sub_dim) const;

    SubMatrixView<T> subMat(dim3 idx, dim3 sub_dim);

    ADD_VALARRAY_OPERATOR(+)
    ADD_VALARRAY_OPERATOR(*)
    ADD_VALARRAY_OPERATOR(-)
    ADD_VALARRAY_OPERATOR(/)

    template<typename T2>
    friend std::ostream& operator<<(std::ostream& os, const SimpleMatrix<T2>& M);

    auto begin() {return std::begin(data_);}
    auto end()   {return std::end(data_);}

    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    size_t getIndex(dim3 dim) const;

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0);
    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const;

    uint dim(size_t idx) const{
        return dim_.data[idx];
    }

    const dim3& dims() const{
        return dim_;
    }

protected:
    dim3 dim_;
    std::valarray<T> data_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const my_cnn::SimpleMatrix<T>& M){
    const T max = M.data_.max();
    const int max_width = ceil(log10(max));

    for (int i = 0; i < M.dim_.x; i++){
        for (int k = 0; k < M.dim_.z; k++){
            os << "[";
            for (int j = 0; j < M.dim_.y; j++){
                os << std::setw(max_width) << M.data_[M.getIndex(i, j, k)] << (j == M.dim_.y - 1 ? "]" : ", ");
            }
            os << "   ";
        }
        os << "\n";
    }

    return os;
}

} // namespace my_cnn
