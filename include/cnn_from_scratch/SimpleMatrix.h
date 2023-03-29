#pragma once

#include <math.h>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>

namespace my_cnn{

struct dim3{
    uint x;
    uint y;
    uint z;

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

    SimpleMatrix<T> operator+(const SimpleMatrix<T>& other){
        if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;
        SimpleMatrix<T> new_mat = *this;
        new_mat.data_ += other.data_;
        return new_mat;
    }

    SimpleMatrix<T> operator+=(const SimpleMatrix<T>& other){
        if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;
        this->data_ += other.data_;
        return *this;
    }

    SimpleMatrix operator*(const SimpleMatrix& other){
        if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;
        SimpleMatrix<T> new_mat = *this;
        new_mat.data_ *= other.data_;
        return new_mat;
    }

    SimpleMatrix operator*=(const SimpleMatrix& other){
        if (dim_ != other.dim_) THROW_SIZE_EXCEPTION;
        SimpleMatrix<T> new_mat = *this;
        this->data_ *= other.data_;
        return new_mat;
    }

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

protected:
    dim3 dim_;
    std::valarray<T> data_;
};

} // namespace my_cnn
