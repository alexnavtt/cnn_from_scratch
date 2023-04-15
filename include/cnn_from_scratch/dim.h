#pragma once

#include <cstring>
#include <ostream>

namespace my_cnn{

template<int N>
struct Dim{
    union{
        struct{
            int x;
            int y;
            int z;
        };
        int data[std::max(N, 3)] = {0};
    };
    static const int n = N;

    Dim() = default;

    Dim(int val) {
        std::fill(data, data + N*sizeof(*data), val);
    }

    template<typename... ValSet>
    Dim(ValSet... vals) {
        static_assert((std::is_integral_v<ValSet> && ...));
        static_assert(sizeof...(vals) <= N);
        size_t i = 0; 
        (void(data[i++] = vals), ...);
    }

    friend std::ostream& operator << (std::ostream& os, const Dim<N>& dim){
        os << "(";
        for (int i = 0; i < n; i++){
            os << dim.data[i];
            if (i != n-1) 
                os << ", ";
            else
                os << ")";
        }
        return os;
    }

    bool operator==(const Dim<N>& other) const{
        bool equal = true;
        for (size_t i = 0; (i < N) && equal; i++){
            equal = equal && data[i] == other.data[i];
        }
        return equal;
    }

    bool operator!=(const Dim<N>& other) const{
        return not (other == *this);
    }

    Dim<N> operator+(const Dim<N>& other) const{
        Dim<N> result;
        for (int i = 0; i < N; i++){
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Dim<N> operator-(const Dim<N>& other) const{
        Dim<N> result;
        for (int i = 0; i < N; i++){
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
};  

typedef Dim<3> dim3;
typedef Dim<2> dim2;

template<int N>
struct DimIterator{
    DimIterator(Dim<N> dim, Dim<N> idx) : dim(dim), idx(idx) {}

    DimIterator operator++(int){
        DimIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    DimIterator& operator++(){
        for (int i = 0; i < Dim<N>::n; i++){
            if ((idx.data[i] + 1 < dim.data[i]) || (i == Dim<N>::n - 1)) {
                idx.data[i]++; 
                break;
            }
            else idx.data[i] = 0;
        }
        return *this;
    }

    bool operator==(const DimIterator& other) const{
        return idx == other.idx && dim == other.dim;
    }

    Dim<N> idx;
    Dim<N> dim;
};

} // namespace my_cnn
