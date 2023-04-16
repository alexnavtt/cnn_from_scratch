#pragma once

#include <math.h>
#include <utility>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <type_traits>

#include "cnn_from_scratch/Matrix/dim.h"
#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/Matrix/SubMatrixView.h"

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

    // Make SubMatrixView a friend
    template <typename MatrixType>
    friend class SubMatrixView;
    
public:

    using type = T;

    /* === Constructors === */

    // Default constructor
    SimpleMatrix() = default;

    // Initial value based constructor
    SimpleMatrix(dim3 dim, T initial_val=T{});

    // Full matrix description constructor
    SimpleMatrix(dim3 dim, std::valarray<T>&& vals);

    // From a gslice_array
    SimpleMatrix(dim3 dim, std::gslice_array<T>&& vals);

    // Type conversion constructor
    template<typename Other, typename = std::enable_if_t<not std::is_same_v<T, Other>>>
    SimpleMatrix(const SimpleMatrix<Other>& M); 

    // Copy constructors
    SimpleMatrix(const SimpleMatrix& M);
    SimpleMatrix(SimpleMatrix<T>&& M);

    /* === Size Checking === */

    // Check against another matrix
    template<typename Other>
    bool sizeCheck(const SimpleMatrix<Other>& other) const noexcept{
        if (other.dim_ == dim_) return true;
        printf("Size mismatch (Matrix): Compared sizes are (%u, %u, %u) for this and (%u, %u, %u) for other\n", 
                dim_.x, dim_.y, dim_.z, other.dim_.x, other.dim_.y, other.dim_.z);
        return false;
    }

    // Check against a matrix view
    template<typename Other>
    bool sizeCheck(const SubMatrixView<Other>& other) const noexcept{
        if (other.dim_ == dim_) return true;
        printf("Size mismatch (Matrix): Compared sizes are (%u, %u, %u) for this and (%u, %u, %u) for other\n", 
                dim_.x, dim_.y, dim_.z, other.dim_.x, other.dim_.y, other.dim_.z);
        return false;
    }

    // Check against a valarray or gslice_array
    template <typename ValarrayLike, std::enable_if_t<
        std::is_convertible_v<
            ValarrayLike, 
            std::valarray<typename ValarrayLike::value_type>>, 
        bool> = true>
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
    SimpleMatrix<T>& operator=(std::valarray<T>&& v);

    // Value setting
    SimpleMatrix<T>& operator=(std::gslice_array<T>&& v);

    // Type conversion
    template<typename Other>
    SimpleMatrix<T>& operator=(const SimpleMatrix<Other>& M);

    void setEntries(std::valarray<T>&& v);

    /* === Indexing === */

    // Get the scalar index into the matrix given a 3d index
    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    size_t getIndex(dim3 dim) const;

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0);
    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    T& operator()(dim3 idx);
    const T& operator()(dim3 idx) const;

    SimpleMatrix<T> subMatCopy(dim3 idx, dim3 sub_dim) const;

    SubMatrixView<SimpleMatrix<T>> subMatView(dim3 idx, dim3 sub_dim);

    SubMatrixView<const SimpleMatrix<T>> subMatView(dim3 idx, dim3 sub_dim) const;

    /* === Arithmetic === */

    ADD_MATRIX_CONST_OPERATOR(+);
    ADD_MATRIX_CONST_OPERATOR(-);
    ADD_MATRIX_CONST_OPERATOR(*);
    ADD_MATRIX_CONST_OPERATOR(/);

    ADD_MATRIX_MODIFYING_OPERATOR(+=);
    ADD_MATRIX_MODIFYING_OPERATOR(-=);
    ADD_MATRIX_MODIFYING_OPERATOR(*=);
    ADD_MATRIX_MODIFYING_OPERATOR(/=);

    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<typename Other::type>>, bool> = true>
    typename std::common_type_t<T, typename Other::type> dot(const Other& M) const;

    template<typename Other>
    SimpleMatrix<typename std::common_type<T, Other>::type> matMul(const SimpleMatrix<Other> M) const;

    /* === Dimension === */

    const dim3& dim() const;
    const dim3& dims() const;
    uint dim(size_t idx) const;

    void reshape(int x, int y, int z);
    void reshape(dim3 new_dim){this->reshape(new_dim.x, new_dim.y, new_dim.z);}

    SimpleMatrix<T> transpose();

    /* === Other Math === */

    bool operator ==(const SimpleMatrix<T>& other) const;

    SimpleMatrix<T> abs() const;

    size_t minIndex() const;
    size_t maxIndex() const;

    SubMatrixView<SimpleMatrix<T>> slices(int idx, int num);
    SubMatrixView<SimpleMatrix<T>> slice(int idx);
    SubMatrixView<const SimpleMatrix<T>> slices(int idx, int num) const;
    SubMatrixView<const SimpleMatrix<T>> slice(int idx) const;

protected:
    dim3 dim_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const SimpleMatrix<T>& M){
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

#include <SubMatrixView.tpp>
#include <SimpleMatrix.tpp>
