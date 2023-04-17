#pragma once

#include <string>
#include <iterator>
#include "cnn_from_scratch/Matrix/MatrixBase.h"
#include "cnn_from_scratch/Matrix/SubMatrixViewIterator.h"

namespace my_cnn{

// Matrix forward declaration
template <typename T>
class SimpleMatrix;

template <typename T>
class SubMatrixView : public MatrixBase {

    // Declare friends
    template<typename Other>
    friend class SimpleMatrix;

    template<typename A>
    friend class MatrixIterator;

public:

    using MatrixType = std::conditional_t<
        std::is_const_v<T>, 
        const SimpleMatrix<typename std::remove_const_t<T>>, 
        SimpleMatrix<typename std::remove_const_t<T>>
    >;
    using type = T;

    SubMatrixView(MatrixType& mat, dim3 start, dim3 dim) : 
    MatrixBase(dim), mat_ptr_(&mat), start_(start) {}

    SubMatrixView(const SubMatrixView<MatrixType>& other_view, dim3 start, dim3 dim) : 
    MatrixBase(dim), mat_ptr_(other_view.mat_ptr_), start_(other_view.start_ + start) {}

    // Indexing - const
    const type& operator()(const dim3& idx) const;
    const type& at(const dim3& idx) const;
    // Indexing - non-const
    type& operator()(const dim3& idx);
    type& at(const dim3& idx);

    // Convert to SimpleMatrix
    operator SimpleMatrix<std::remove_const_t<T>>() const;
    SimpleMatrix<std::remove_const_t<T>> matrix() const;

    // Assign to a contained type
    template<typename Other, 
            std::enable_if_t<std::is_convertible_v<Other, type>, bool> = true >
    SubMatrixView<T>& operator=(const Other& o);

    // Assign to a matrix-like object
    template<typename Other, 
            std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true >
    SubMatrixView<T>& operator=(const Other& o);

    // Iterators
    auto begin() {
        return MatrixIterator<SubMatrixView<T>>(this, dim3(0, 0, 0));
    }
    auto end() {
        MatrixIterator<SubMatrixView<T>> it(this, dim_ - dim3(1));
        return ++it;
    }
    auto begin() const {
        return MatrixIterator<const SubMatrixView<T>>(this, dim3(0, 0, 0));
    }
    auto end() const {
        MatrixIterator<const SubMatrixView<T>> it(this, dim_ - dim3(1));
        return ++it;
    }

private:
    dim3 start_;
    MatrixType* mat_ptr_;
};

} // namespace my_cnn
