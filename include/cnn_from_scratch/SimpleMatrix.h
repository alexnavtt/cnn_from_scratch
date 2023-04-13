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
        auto& out_varr = static_cast<std::valarray<T>&>(out); \
        auto& curr_varr = static_cast<const std::valarray<T>&>(*this);  \
        out_varr = curr_varr op other; \
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

    /* === Constructors === */

    // Default constructor
    SimpleMatrix() = default;

    // Initial value based constructor
    SimpleMatrix(dim3 dim, T initial_val=T{}):
    std::valarray<T>(initial_val, dim.x*dim.y*dim.z),
    dim_(dim)
    {}

    // Full matrix description constructor
    SimpleMatrix(dim3 dim, std::valarray<T>&& vals):
    std::valarray<T>(dim.x*dim.y*dim.z),
    dim_(dim)
    {
        setEntries(std::forward<std::valarray<T>>(vals));
    }

    // From a gslice_array
    SimpleMatrix(dim3 dim, std::gslice_array<T>&& vals):
    std::valarray<T>(vals),
    dim_(dim)
    {}

    // Type conversion constructor
    template<typename Other>
    SimpleMatrix(const SimpleMatrix<Other>& M) : 
    std::valarray<T>(M.dim_.x*M.dim_.y*M.dim_.z),
    dim_(M.dim_)
    {
        std::copy(std::begin(M), std::end(M), std::begin(*this));
    }

    /* === Size Checking === */

    // Check against another matrix
    template<typename Other>
    bool sizeCheck(const SimpleMatrix<Other>& other) const noexcept{
        if (other.dim_ == dim_) return true;
        printf("Size mismatch (Matrix): Compared sizes are (%u, %u, %u) for this and (%u, %u, %u) for other\n", 
                dim_.x, dim_.y, dim_.z, other.dim_.x, other.dim_.y, other.dim_.z);
        return false;
    }

    // Check against a valarray or gslice_array
    template<typename ValarrayLike,
    std::enable_if_t<std::is_convertible<ValarrayLike, std::valarray<typename ValarrayLike::value_type>>::value, bool> = true>
    bool sizeCheck(const ValarrayLike& v) const {
        using U = typename ValarrayLike::value_type;
        if(static_cast<std::valarray<U>>(v).size() == this->size()) return true;
        printf("Size mismatch (Valarray/gslice_array): Compared sizes are (%u, %u, %u) (i.e. size %zd) for this and %zd for other\n", 
                dim_.x, dim_.y, dim_.z, this->size(), static_cast<std::valarray<U>>(v).size());
        return false;
    }

    // Literal value
    template<typename Other, std::enable_if_t<std::is_arithmetic<Other>::value, bool> = true>
    bool sizeCheck(const Other& v) const noexcept{
        return true;
    }

    /* == Assignment === */

    // Default
    SimpleMatrix<T>& operator=(const SimpleMatrix<T>& M) = default;

    // Value setting
    SimpleMatrix<T>& operator=(std::valarray<T>&& v){
        if (v.size() == this->size())
            static_cast<std::valarray<T>&>(*this) = std::forward<std::valarray<T>>(v);
        else
            throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
        return *this;
    }

    // Value setting
    SimpleMatrix<T>& operator=(std::gslice_array<T>&& v){
        std::valarray<T> arr(v);
        if (arr.size() == this->size())
            static_cast<std::valarray<T>&>(*this) = arr;
        else
            throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
        return *this;
    }

    // Type conversion
    template<typename Other>
    SimpleMatrix<T>& operator=(const SimpleMatrix<Other>& M){
        dim_ = M.dim_;
        this->resize(dim_.x*dim_.y*dim_.z);
        std::copy(std::begin(M), std::end(M), std::begin(*this));
        return *this;
    }

    /* === Indexing === */

    // Get the scalar index into the matrix given a 3d index
    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    size_t getIndex(dim3 dim) const;

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0);
    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    T& operator()(dim3 idx);
    const T& operator()(dim3 idx) const;

    SimpleMatrix<T> subMatCopy(dim3 idx, dim3 sub_dim) const;

    std::gslice subMatIdx(dim3 start, dim3 size) const{
        return std::gslice(
            getIndex(start),
            {size.z, size.y, size.x},
            {dim_.y*dim_.x, dim_.x, 1}
        );
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

    /* === Arithmetic === */

    ADD_MATRIX_CONST_OPERATOR(+);
    ADD_MATRIX_CONST_OPERATOR(-);
    ADD_MATRIX_CONST_OPERATOR(*);
    ADD_MATRIX_CONST_OPERATOR(/);

    ADD_MATRIX_MODIFYING_OPERATOR(+=);
    ADD_MATRIX_MODIFYING_OPERATOR(-=);
    ADD_MATRIX_MODIFYING_OPERATOR(*=);
    ADD_MATRIX_MODIFYING_OPERATOR(/=);

    /* === Dimension === */

    uint dim(size_t idx) const{
        return dim_.data[idx];
    }

    const dim3& dims() const{
        return dim_;
    }

    /* === Other Math === */

    bool operator ==(const SimpleMatrix<T>& other) const{
        if (dim_ != other.dim_) return false;
        for (size_t i = 0; i < this->size(); i++){
            if (this->operator[](i) != other[i]) return false;
        }
        return true;
    }

    SimpleMatrix<T> abs() const {
        if constexpr (std::is_unsigned_v<T>)
            return *this;
        else{
            SimpleMatrix<T> out(dim_);
            return out = std::abs(*this);
        }
    }

    std::gslice slice(unsigned idx) const{
        return std::gslice(idx*dim_.y*dim_.x, {1, dim_.y, dim_.x}, {dim_.x*dim_.y, dim_.x, 1});
    }

    std::gslice slices(unsigned idx, unsigned num, unsigned stride = 1) const{
        return std::gslice(idx*dim_.y*dim_.x, {num, dim_.y, dim_.x}, {stride*dim_.x*dim_.y, dim_.x, 1});
    }

    SimpleMatrix<T> sliceCopy(unsigned idx) const{
        SimpleMatrix<T> out(dim_.slice());
        out = this->operator[](slice(idx));
        return out;
    }

protected:
    dim3 dim_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const my_cnn::SimpleMatrix<T>& M){
    int default_precision = std::cout.precision();
    const T max = abs(M).max();
    int max_width = ceil(log10(max));
    
    if constexpr (std::is_floating_point_v<T>){
        std::cout << std::setprecision(2) << std::fixed;
        max_width += 3;
    }

    for (int i = 0; i < M.dim(0); i++){
        for (int k = 0; k < M.dim(2); k++){
            os << "[";
            for (int j = 0; j < M.dim(1); j++){
                const auto& val = M(i, j, k);
                os << (val < 0 ? "-" : " ") << std::setw(max_width) << abs(val) << (j == M.dim(1) - 1 ? "]" : ",");
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
