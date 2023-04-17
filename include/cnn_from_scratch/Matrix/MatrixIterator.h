#pragma once

#include <iterator>
#include <cnn_from_scratch/Matrix/dim.h>

namespace my_cnn{

template<typename>
class SimpleMatrix;

struct empty_{
    empty_(int){};
};

template<typename MatrixType>
class MatrixIterator{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = typename MatrixType::type;
    using pointer = std::conditional_t<std::is_const_v<MatrixType> || std::is_const_v<typename MatrixType::type>, const value_type*, value_type*>;
    using reference = decltype(std::declval<MatrixType>().operator()(std::declval<dim3>()));

    MatrixIterator(MatrixType* parent, dim3 idx) : 
    parent_(parent), 
    dim_it_(parent->dim(), idx),
    scalar_idx_(idx.z*dim_it_.dim.x*dim_it_.dim.y + idx.y*dim_it_.dim.x + idx.x) {}

    reference operator*() {
        if constexpr (is_pure_matrix)
            return parent_->values_[scalar_idx_];
        else
            return parent_->operator()(dim_it_.idx);
    }
    pointer operator->() {return &this->operator*();}

    const dim3& idx(){
        return dim_it_.idx;
    }

    // Post-increment
    MatrixIterator<MatrixType> operator++(int){
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    MatrixIterator<MatrixType>& operator++(){
        dim_it_++;
        if constexpr(is_pure_matrix)
            scalar_idx_++;
        return *this;
    }

    // Difference of two iterators
    difference_type operator-(const MatrixIterator& other){
        dim3 diff = dim_it_.idx - other.dim_it_.idx;
        return (diff.z * dim_it_.dim.x * dim_it_.dim.y) + (diff.y * dim_it_.dim.x) + diff.x;
    }

    bool operator==(const MatrixIterator& other){
        return dim_it_ == other.dim_it_ && parent_ == other.parent_;
    }
    bool operator!=(const MatrixIterator& other){
        return not (*this == other);
    }

private:
    std::conditional_t<std::is_const_v<MatrixType>, const MatrixType*, MatrixType*> parent_;
    DimIterator<3> dim_it_;
    static constexpr bool is_pure_matrix = std::is_same_v<typename std::remove_const_t<MatrixType>, SimpleMatrix<typename MatrixType::type>>;
    std::conditional_t<is_pure_matrix, size_t, empty_> scalar_idx_;
};
    
} // namespace my_cnn
