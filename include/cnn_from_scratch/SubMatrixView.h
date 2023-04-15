#pragma once

#include <string>
#include <iterator>
#include "cnn_from_scratch/dim.h"
#include "cnn_from_scratch/SubMatrixViewIterator.h"

namespace my_cnn{

// Matrix forward declaration
template <typename T>
class SimpleMatrix;

template <typename T, typename MatrixType = SimpleMatrix<T>>
class SubMatrixView {

    // Declare friends
    template<typename Other>
    friend class SimpleMatrix;

    template<typename A, typename B>
    friend class SubMatrixIterator;

    template<typename A, typename B>
    friend class SubMatrixConstIterator;

public:

    SubMatrixView(MatrixType& mat, dim3 start, dim3 dim) : mat_ptr_(&mat), start_(start), dim_(dim) {}

    size_t size() const noexcept {return dim_.x * dim_.y * dim_.z;}

    // Indexing
    const T& operator()(dim3 idx) const;
    const T& at(dim3 idx) const;
    decltype(auto) operator()(dim3 idx){return const_cast<T&>(const_cast<const SubMatrixView<T>&>(*this).operator()(idx));}
    decltype(auto) at(dim3 idx){return const_cast<T&>(const_cast<const SubMatrixView<T>&>(*this).operator()(idx));}
    

    // Convert to SimpleMatrix
    operator SimpleMatrix<T>() const;

    // Assign to a contained type
    template<typename Other, 
            std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool> = true >
    SubMatrixView<T, MatrixType>& operator=(const Other& o);

    // Assign to a matrix or matrix view
    template<typename Other, 
            std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<T>> && not std::is_const_v<MatrixType>, bool> = true >
    SubMatrixView<T, MatrixType>& operator=(const Other& o);

    // All operator functions look the same, so we'll assign them via macro
    #define ADD_MODIFYING_OPERATOR(op)                                                                                             \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool> = true >   \
    SubMatrixView<T, MatrixType> operator op(const Other& o);                                                                      \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool> = true>    \
    SubMatrixView<T, MatrixType> operator op(const SimpleMatrix<Other>& o);                                                        \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool> = true>    \
    SubMatrixView<T, MatrixType> operator op(const SubMatrixView<Other>& o);  

    ADD_MODIFYING_OPERATOR(+=);
    ADD_MODIFYING_OPERATOR(-=);
    ADD_MODIFYING_OPERATOR(*=);
    ADD_MODIFYING_OPERATOR(/=);

    #define ADD_OUTPUT_OPERATOR(op)                                                                         \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T>, bool> = true >               \
    SimpleMatrix<typename std::common_type_t<T, Other>> operator op(const Other& o) const;                  \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<T>>, bool> = true>  \
    SimpleMatrix<typename std::common_type_t<T, Other>> operator op(const Other& o) const;  

    ADD_OUTPUT_OPERATOR(+);
    ADD_OUTPUT_OPERATOR(-);
    ADD_OUTPUT_OPERATOR(*);
    ADD_OUTPUT_OPERATOR(/);

    // Iterators
    SubMatrixIterator<T, MatrixType> begin() {
        return SubMatrixIterator<T, MatrixType>(this, dim3(0, 0, 0));
    }
    SubMatrixIterator<T, MatrixType> end() {
        SubMatrixIterator<T, MatrixType> it(this, dim_ - dim3(1));
        return ++it;
    }
    SubMatrixConstIterator<T, MatrixType> begin() const {
        return SubMatrixConstIterator<T, MatrixType>(this, dim3(0, 0, 0));
    }
    SubMatrixConstIterator<T, MatrixType> end() const {
        SubMatrixConstIterator<T, MatrixType> it(this, dim_ - dim3(1));
        return ++it;
    }

private:
    dim3 start_;
    dim3 dim_;
    MatrixType* mat_ptr_;
};

} // namespace my_cnn
