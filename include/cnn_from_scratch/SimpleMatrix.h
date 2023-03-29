#pragma once

#include <math.h>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <type_traits>

namespace my_cnn{

struct dim3{
    union{
        struct{
            uint x;
            uint y;
            uint z;
        };
        uint data[3];
    };

    bool operator==(const dim3& other) const{return (x == other.x) && (y == other.y) && (z == other.z);}
    bool operator!=(const dim3& other) const{return not (other == *this);}
};

class MatrixSizeException : public std::exception{
public:
    MatrixSizeException(std::string msg = "") : msg_{msg} {}
    const char* what() const noexcept {return msg_.c_str();}
private:
    std::string msg_;
};

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

    SimpleMatrix(const SimpleMatrix<T>& M, dim3 idx, dim3 sub_dim) :
    dim_(sub_dim),
    data_(M.data_[std::gslice(
        M.getIndex(idx.x, idx.y, idx.z), 
        {sub_dim.z, sub_dim.y, sub_dim.x}, 
        {M.dim_.y*M.dim_.x, M.dim_.x, 1}
    )])
    {}

    ADD_VALARRAY_OPERATOR(+)
    ADD_VALARRAY_OPERATOR(*)
    ADD_VALARRAY_OPERATOR(-)
    ADD_VALARRAY_OPERATOR(/)

    friend std::ostream& operator<<(std::ostream& os, const SimpleMatrix<T>& M){
        T max = M.data_.max();
        int max_width = ceil(log10(max));

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

    auto begin() {return std::begin(data_);}
    auto end() {return std::end(data_);}

    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const{
        return z_idx * dim_.y * dim_.x + y_idx * dim_.x + x_idx;
    }

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0){
        return data_[getIndex(x_idx, y_idx, z_idx)];
    }

    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const {
        return data_[getIndex(x_idx, y_idx, z_idx)];
    }

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

} // namespace my_cnn
