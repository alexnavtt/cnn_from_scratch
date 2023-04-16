#pragma once

#include <iterator>
#include <cnn_from_scratch/Matrix/dim.h>

namespace my_cnn{

template<typename MatrixType>
class SubMatrixView;

template<typename MatrixType>
class SubMatrixIterator{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = typename MatrixType::type;
    using pointer = std::conditional_t<std::is_const_v<MatrixType>, const value_type*, value_type*>;
    using reference = std::conditional_t<std::is_const_v<MatrixType>, const value_type&, value_type&>;

    SubMatrixIterator(SubMatrixView<MatrixType>* parent, dim3 idx) : parent_(parent), dim_it_(parent->dim_, idx) {}
    decltype(auto) operator*() {return parent_->operator()(dim_it_.idx);}
    decltype(auto) operator*() const {return parent_->operator()(dim_it_.idx);}
    pointer operator->() {return &parent_->operator()(dim_it_.idx);}

    // Post-increment
    SubMatrixIterator<MatrixType> operator++(int){
        SubMatrixIterator<MatrixType> tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    SubMatrixIterator<MatrixType>& operator++(){
        dim_it_++;
        return *this;
    }

    difference_type operator-(const SubMatrixIterator& other){
        dim3 diff = dim_it_.idx - other.dim_it_.dim;
        return (diff.z * dim_it_.dim.x * dim_it_.dim.y) + (diff.y * dim_it_.dim.x) + diff.x;
    }

    friend bool operator==(const SubMatrixIterator<MatrixType>& a, const SubMatrixIterator<MatrixType>& b){
        return a.dim_it_ == b.dim_it_ && a.parent_ == b.parent_;
    }
    friend bool operator!=(const SubMatrixIterator<MatrixType>& a, const SubMatrixIterator<MatrixType>& b){return not (a == b);}

private:
    SubMatrixView<MatrixType>* parent_;
    DimIterator<3> dim_it_;
};

template<typename MatrixType>
class SubMatrixConstIterator{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = typename MatrixType::type;
    using pointer = std::conditional_t<std::is_const_v<MatrixType>, const value_type*, value_type*>;
    using reference = std::conditional_t<std::is_const_v<MatrixType>, const value_type&, value_type&>;

    SubMatrixConstIterator(const SubMatrixView<MatrixType>* parent, dim3 idx) : parent_(parent), dim_it_(parent->dim_, idx) {}
    decltype(auto) operator*() {return parent_->operator()(dim_it_.idx);}
    decltype(auto) operator*() const {return parent_->operator()(dim_it_.idx);}
    pointer operator->() {return &parent_->operator()(dim_it_.idx);}

    // Post-increment
    SubMatrixConstIterator<MatrixType> operator++(int){
        SubMatrixConstIterator<MatrixType> tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    SubMatrixConstIterator<MatrixType>& operator++(){
        dim_it_++;
        return *this;
    }

    difference_type operator-(const SubMatrixConstIterator<MatrixType>& other){
        dim3 diff = dim_it_.idx - other.dim_it_.idx;
        return (diff.z * dim_it_.dim.x * dim_it_.dim.y) + (diff.y * dim_it_.dim.x) + diff.x;
    }

    friend bool operator==(const SubMatrixConstIterator<MatrixType>& a, const SubMatrixConstIterator<MatrixType>& b){
        return a.dim_it_ == b.dim_it_ && a.parent_ == b.parent_;
    }
    friend bool operator!=(const SubMatrixConstIterator<MatrixType>& a, const SubMatrixConstIterator<MatrixType>& b){return not (a == b);}

private:
    const SubMatrixView<MatrixType>* parent_;
    DimIterator<3> dim_it_;
};
    
} // namespace my_cnn
