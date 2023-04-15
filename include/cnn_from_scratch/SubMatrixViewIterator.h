#pragma once

#include <iterator>
#include <cnn_from_scratch/dim.h>

namespace my_cnn{

template<typename T, typename MatrixType>
class SubMatrixView;

template<typename T, typename MatrixType>
class SubMatrixIterator{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::conditional_t<std::is_const_v<MatrixType>, const T*, T*>;
    using reference = std::conditional_t<std::is_const_v<MatrixType>, const T&, T&>;

    SubMatrixIterator(SubMatrixView<T, MatrixType>* parent, dim3 idx) : parent_(parent), dim_(parent->dim_), idx_(idx) {}
    decltype(auto) operator*() {return parent_->operator()(idx_);}
    decltype(auto) operator*() const {return parent_->operator()(idx_);}
    pointer operator->() {return &parent_->operator()(idx_);}

    // Post-increment
    SubMatrixIterator operator++(int){
        SubMatrixIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    SubMatrixIterator& operator++(){
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

    friend bool operator==(const SubMatrixIterator& a, const SubMatrixIterator& b){
        return (a.dim_ == b.dim_) && (a.idx_ == b.idx_);
    }
    friend bool operator!=(const SubMatrixIterator& a, const SubMatrixIterator& b){return not (a == b);}

private:
    SubMatrixView<T, MatrixType>* parent_;
    dim3 dim_;
    dim3 idx_;
};

template<typename T, typename MatrixType>
class SubMatrixConstIterator{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = const T&;

    SubMatrixConstIterator(const SubMatrixView<T, MatrixType>* parent, dim3 idx) : parent_(parent), dim_(parent->dim_), idx_(idx) {}
    reference operator*() const {return (*parent_)(idx_);}
    pointer operator->() {return &parent_->operator()(idx_);}

    // Post-increment
    SubMatrixConstIterator operator++(int){
        SubMatrixConstIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    // Pre-increment
    SubMatrixConstIterator& operator++(){
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

    difference_type operator-(const SubMatrixConstIterator& other){
        dim3 diff = idx_ - other.idx_;
        return diff.z * dim_.x * dim_.y + diff.y * dim_.x + diff.x;
    }

    friend bool operator==(const SubMatrixConstIterator& a, const SubMatrixConstIterator& b){
        return (a.dim_ == b.dim_) && (a.idx_ == b.idx_);
    }
    friend bool operator!=(const SubMatrixConstIterator& a, const SubMatrixConstIterator& b){return not (a == b);}

private:
    const SubMatrixView<T, MatrixType>* parent_;
    dim3 dim_;
    dim3 idx_;
};
    
} // namespace my_cnn
