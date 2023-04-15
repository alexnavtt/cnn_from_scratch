#pragma once

#include <math.h>
#include <utility>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <type_traits>

#include "cnn_from_scratch/dim.h"
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

    // Make SubMatrixView a friend
    template <typename MatrixType>
    friend class SubMatrixView;
    
public:

    using type = T;

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
    template<typename Other, typename = std::enable_if_t<not std::is_same_v<T, Other>>>
    SimpleMatrix(const SimpleMatrix<Other>& M) : 
    std::valarray<T>(M.dim_.x*M.dim_.y*M.dim_.z),
    dim_(M.dim_)
    {
        std::copy(std::begin(M), std::end(M), std::begin(*this));
    }

    SimpleMatrix(const SimpleMatrix& M) : std::valarray<T>(M) , dim_(M.dim_) {}
    SimpleMatrix(SimpleMatrix<T>&& M) : std::valarray<T>(std::forward<std::valarray<T>>(M)), dim_(M.dim_) {}

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
        std::valarray<T>::resize(dim_.x*dim_.y*dim_.z);
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

    SubMatrixView<SimpleMatrix<T>> subMatView(dim3 idx, dim3 sub_dim){
        return SubMatrixView(*this, idx, sub_dim);
    }

    SubMatrixView<const SimpleMatrix<T>> subMatView(dim3 idx, dim3 sub_dim) const{
        return SubMatrixView(*this, idx, sub_dim);
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

    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<typename Other::type>>, bool> = true>
    typename std::common_type_t<T, typename Other::type> dot(const Other& M) const{
        if (this->size() != M.size()){
            throw MatrixSizeException("Cannot get the dot product A.B, size mismatch. A has " 
                + std::to_string(this->size()) + " elements and B has " + std::to_string(M.size()) + " elements");
        }

        if constexpr (std::is_same_v<Other, SimpleMatrix<T>>){
            return (static_cast<const std::valarray<T>&>(*this) * static_cast<const std::valarray<T>&>(M)).sum();
        }else{
            std::common_type_t<T, typename Other::type> sum{};
            auto this_it = std::begin(*this);
            auto other_it = std::begin(M);
            for (; this_it != std::end(*this); this_it++, other_it++){
                sum += *this_it * *other_it;
            }
            return sum;
        }
    }

    template<typename Other>
    SimpleMatrix<typename std::common_type<T, Other>::type> matMul(const SimpleMatrix<Other> M) const{
        if (dim_.z != M.dim_.z){
            std::cout << "Matrix multiply error: Differing number of layers. ";
            std::cout << "This has " << dim_.z << " layers and the other has " << M.dim_.z << "\n";
            throw MatrixSizeException("Multiplication layer mismatch");
        }else if(dim_.y != M.dim_.x){
            std::cout << "Matrix multiply error: Incompatible matrix dimensions ";
            std::cout << dim_ << " and " << M.dim_ << "\n";
            throw MatrixSizeException("Multiplication dimension mismatch");
        }

        SimpleMatrix<typename std::common_type<T, Other>::type> out({dim_.x, M.dim_.y, dim_.z});
        for (uint row = 0; row < out.dim_.x; row++){
            for (uint col = 0; col < out.dim_.y; col++){
                for (uint layer = 0; layer < dim_.z; layer++){

                    out(row, col, layer) = 
                        this->subMatCopy({row, 0, layer}, {1, dim_.y, 1})
                            .dot(M.subMatView({0, col, layer}, {M.dim_.x, 1, 1}));

                }
            }
        }

        return out;
    }

    /* === Dimension === */

    uint dim(size_t idx) const{
        return dim_.data[idx];
    }

    const dim3& dims() const{
        return dim_;
    }

    void reshape(unsigned x, unsigned y, unsigned z){
        if(x*y*z != dim_.x*dim_.y*dim_.z){
            std::stringstream ss;
            ss << "Cannot transform matrix from size " << dim_ << " to size " << dim3(x,y,z) << "\n";
            throw MatrixTransformException(ss.str());
        }
        dim_ = dim3(x, y, z);
    }

    void reshape(dim3 new_dim){
        this->reshape(new_dim.x, new_dim.y, new_dim.z);
    }

    SimpleMatrix<T> transpose(){
        dim3 new_dim(dim_.y, dim_.x, dim_.z);
        SimpleMatrix<T> output(new_dim);
        static_cast<std::valarray<T>&>(output) = (*this)[
            std::gslice(
                0, 
                {dim_.z, dim_.x, dim_.y},
                {dim_.x*dim_.y, 1, dim_.x}
            )
        ];
        return output;
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

    size_t minIndex() const{
        return std::distance(std::begin(*this), std::min_element(std::begin(*this), std::end(*this)));
    }

    size_t maxIndex() const{
        return std::distance(std::begin(*this), std::max_element(std::begin(*this), std::end(*this)));
    }

    SubMatrixView<SimpleMatrix<T>> slices(unsigned idx, unsigned num){
        return SubMatrixView(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
    }

    SubMatrixView<SimpleMatrix<T>> slice(unsigned idx) {
        return slices(idx, 1);
    }

    SubMatrixView<const SimpleMatrix<T>> slices(unsigned idx, unsigned num) const{
        return SubMatrixView(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
    }

    SubMatrixView<const SimpleMatrix<T>> slice(unsigned idx) const {
        return slices(idx, 1);
    }

    SimpleMatrix<T> sliceCopy(unsigned idx) const{
        SimpleMatrix<T> out = slices(idx, 1);
        return out;
    }

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
