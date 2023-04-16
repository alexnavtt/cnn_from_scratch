#pragma once

#include <string>
#include <iterator>
#include "cnn_from_scratch/Matrix/MatrixBase.h"
#include "cnn_from_scratch/Matrix/SubMatrixViewIterator.h"

namespace my_cnn{

// Matrix forward declaration
template <typename T>
class SimpleMatrix;

template <typename MatrixType>
class SubMatrixView : public MatrixBase {

    // Declare friends
    template<typename Other>
    friend class SimpleMatrix;

    template<typename A>
    friend class SubMatrixIterator;

    template<typename A>
    friend class SubMatrixConstIterator;

public:

    using type = typename MatrixType::type;

    SubMatrixView(MatrixType& mat, dim3 start, dim3 dim) : 
    MatrixBase(dim), mat_ptr_(&mat), start_(start) {}

    SubMatrixView(const SubMatrixView<MatrixType>& other_view, dim3 start, dim3 dim) : 
    MatrixBase(dim), mat_ptr_(other_view.mat_ptr_), start_(other_view.start_ + start) {}

    // Indexing - const
    const type& operator()(dim3 idx) const;
    const type& at(dim3 idx) const;
    // Indexing - non-const using const_cast
    type& operator()(dim3 idx){return const_cast<type&>(const_cast<const SubMatrixView<MatrixType>&>(*this).operator()(idx));}
    type& at(dim3 idx){return const_cast<type&>(const_cast<const SubMatrixView<MatrixType>&>(*this).operator()(idx));}

    // Convert to SimpleMatrix
    operator SimpleMatrix<type>() const;
    SimpleMatrix<type> matrix() const;

    // Assign to a contained type
    template<typename Other, 
            std::enable_if_t<std::is_convertible_v<Other, type>, bool> = true >
    SubMatrixView<MatrixType>& operator=(const Other& o);

    // Assign to a matrix-like object
    template<typename Other, 
            std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true >
    SubMatrixView<MatrixType>& operator=(const Other& o);

    // All operator functions look the same, so we'll assign them via macro
    #define ADD_MODIFYING_OPERATOR(op)                                                                                             \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type> && not std::is_const_v<MatrixType>, bool> = true >   \
    SubMatrixView<MatrixType>& operator op(const Other& o);                                                                      \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type> && not std::is_const_v<MatrixType>, bool> = true>    \
    SubMatrixView<MatrixType>& operator op(const SimpleMatrix<Other>& o);                                                        \
    template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type> && not std::is_const_v<MatrixType>, bool> = true>    \
    SubMatrixView<MatrixType>& operator op(const SubMatrixView<Other>& o);  

    ADD_MODIFYING_OPERATOR(+=);
    ADD_MODIFYING_OPERATOR(-=);
    ADD_MODIFYING_OPERATOR(*=);
    ADD_MODIFYING_OPERATOR(/=);

    // Iterators
    SubMatrixIterator<MatrixType> begin() {
        return SubMatrixIterator<MatrixType>(this, dim3(0, 0, 0));
    }
    SubMatrixIterator<MatrixType> end() {
        SubMatrixIterator<MatrixType> it(this, dim_ - dim3(1));
        return ++it;
    }
    SubMatrixConstIterator<MatrixType> begin() const {
        return SubMatrixConstIterator<MatrixType>(this, dim3(0, 0, 0));
    }
    SubMatrixConstIterator<MatrixType> end() const {
        SubMatrixConstIterator<MatrixType> it(this, dim_ - dim3(1));
        return ++it;
    }

private:
    dim3 start_;
    MatrixType* mat_ptr_;
};

} // namespace my_cnn
