#pragma once

#include <iterator>
#include <cnn_from_scratch/dim.h>

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

    SubMatrixIterator(SubMatrixView<MatrixType>* parent, dim3 idx) : parent_(parent), dim_(parent->dim_), idx_(idx) {}
    decltype(auto) operator*() {return parent_->operator()(idx_);}
    decltype(auto) operator*() const {return parent_->operator()(idx_);}
    pointer operator->() {return &parent_->operator()(idx_);}

    // Post-increment
    SubMatrixIterator<MatrixType> operator++(int){
        SubMatrixIterator<MatrixType> tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    SubMatrixIterator<MatrixType>& operator++(){
        if (idx_.x + 1 < dim_.x) idx_.x++;
        else if (idx_.y + 1 < dim_.y){
            idx_.x = 0;
            idx_.y++;
        }else{
            idx_.x = 0;
            idx_.y = 0;
            idx_.z++;
        }
        return *this;
    }

    difference_type operator-(const SubMatrixIterator& other){
        dim3 diff = idx_ - other.idx_;
        return diff.z * dim_.x * dim_.y + diff.y * dim_.x + diff.x;
    }

    friend bool operator==(const SubMatrixIterator<MatrixType>& a, const SubMatrixIterator<MatrixType>& b){
        return (a.dim_ == b.dim_) && (a.idx_ == b.idx_);
    }
    friend bool operator!=(const SubMatrixIterator<MatrixType>& a, const SubMatrixIterator<MatrixType>& b){return not (a == b);}

private:
    SubMatrixView<MatrixType>* parent_;
    dim3 dim_;
    dim3 idx_;
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
    DimIterator dim_it_;
};
    
} // namespace my_cnn
