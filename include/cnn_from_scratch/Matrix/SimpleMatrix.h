#pragma once

#include <math.h>
#include <utility>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <type_traits>

#include "cnn_from_scratch/exceptions.h"
#include "cnn_from_scratch/Matrix/MatrixBase.h"
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

template<typename T>  
class SimpleMatrix : public MatrixBase{
    // Make all temlplates of SimpleMatrix friends
    template <typename Other>
    friend class SimpleMatrix;

    // Make SubMatrixView a friend
    template <typename MatrixType>
    friend class SubMatrixView;

    // Make MatrixIterator a friend
    template<typename MatrixType>
    friend class MatrixIterator;
    
public:

    using type = T;

    /* === Constructors === */

    // Default constructor
    SimpleMatrix() = default;

    // From a Matrix-like object
    template<typename MatrixType, std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType>, bool> = true>
    SimpleMatrix(const MatrixType& M);

    // Initial value based constructor
    SimpleMatrix(dim3 dim, T initial_val=T{});

    // Full matrix description constructor
    SimpleMatrix(dim3 dim, std::vector<T>&& vals);

    // Type conversion constructor
    template<typename Other, std::enable_if_t<not std::is_same_v<T, Other>, bool> = true>
    SimpleMatrix(const SimpleMatrix<Other>& M); 

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

    // Check against a vector
    template<typename Other>
    bool sizeCheck(const std::vector<Other>&& v) const {
        if(v.size() == this->values_.size()) return true;
        printf("Size mismatch (Valarray/gslice_array): Compared sizes are (%u, %u, %u) (i.e. size %zd) for this and %zd for other\n", 
                dim_.x, dim_.y, dim_.z, this->size(), v.size());
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
    SimpleMatrix<T>& operator=(std::vector<T>&& v);

    // Type conversion
    template<typename Other>
    SimpleMatrix<T>& operator=(const SimpleMatrix<Other>& M);

    void setEntries(std::vector<T>&& v);

    /* === Indexing === */

    // Get the scalar index into the matrix given a 3d index
    size_t getIndex(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    size_t getIndex(dim3 dim) const;

    T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0);
    const T& operator()(size_t x_idx, size_t y_idx, size_t z_idx=0) const;
    T& operator()(dim3 idx);
    const T& operator()(dim3 idx) const;

    T& operator[](size_t idx){return values_[idx];}
    const T& operator[](size_t idx) const{return values_[idx];}

    SimpleMatrix<T> subMatCopy(dim3 idx, dim3 sub_dim) const;

    SubMatrixView<T> subMatView(dim3 idx, dim3 sub_dim);
    SubMatrixView<const T> subMatView(dim3 idx, dim3 sub_dim) const;

    auto begin() {return MatrixIterator<SimpleMatrix<T>>(this, {0, 0, 0});}
    auto end() {return MatrixIterator<SimpleMatrix<T>>(this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const SimpleMatrix<T>>(this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const SimpleMatrix<T>>(this, {0, 0, dim_.z});}

    /* === Arithmetic === */

    template<typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
    typename std::common_type_t<T, typename Other::type> dot(const Other& M) const;

    template<typename Other>
    SimpleMatrix<typename std::common_type<T, Other>::type> matMul(const SimpleMatrix<Other> M) const;

    /* === Dimension === */

    const dim3& dims() const;
    using MatrixBase::dim;
    uint dim(size_t idx) const;

    void reshape(int x, int y, int z);
    void reshape(dim3 new_dim){this->reshape(new_dim.x, new_dim.y, new_dim.z);}

    SimpleMatrix<T> transpose();

    /* === Other Math === */

    SimpleMatrix<T> abs() const;

    size_t minIndex() const;
    size_t maxIndex() const;

    SubMatrixView<T> slices(int idx, int num);
    SubMatrixView<T> slice(int idx);
    SubMatrixView<T> slices(int idx, int num) const;
    SubMatrixView<T> slice(int idx) const;
private:
    std::vector<T> values_;
};

template<typename MatrixType, std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType>, bool> = true>
std::ostream& operator<<(std::ostream& os, const MatrixType& M){
    using T = typename MatrixType::type;

    int default_precision = std::cout.precision();
    int max_width = 4;
    
    if constexpr (std::is_floating_point_v<T>){
        std::cout << std::setprecision(2) << std::fixed;
        max_width += 3;
    }

    os << "\n";
    for (int i = 0; i < M.dim().x; i++){
        for (int k = 0; k < M.dim().z; k++){
            os << "[";
            for (int j = 0; j < M.dim().y; j++){
                const T& val = M({i, j, k});
                os << (val < 0 ? "-" : " ") << std::setw(max_width) << abs(val) << (j == M.dim().y - 1 ? "]" : ",");
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

#include <cnn_from_scratch/Matrix/MatrixMath.h>
#include <SubMatrixView.tpp>
#include <SimpleMatrix.tpp>
