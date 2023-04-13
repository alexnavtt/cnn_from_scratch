#pragma once

#include <valarray>

namespace my_cnn{
    
template <typename T>
class SubMatrixView : public std::gslice_array<T>{
public:
    SubMatrixView(std::valarray<T>& arr, const std::gslice& g) :
    std::gslice_array<T>(arr[g]),
    g_(g)
    {
        size_ = 1;
        for (const auto& val : g.size()){
            size_ *= val;
        }
    }

    using std::gslice_array<T>::operator =;
    using std::gslice_array<T>::operator +=;
    using std::gslice_array<T>::operator -=;
    using std::gslice_array<T>::operator *=;
    using std::gslice_array<T>::operator /=;

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    SubMatrixView<T> operator+=(const T2& val){
        static_cast<std::gslice_array<T>&>(*this) += std::valarray<T>((T)(val), size_);
        return *this;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    SubMatrixView<T> operator-=(const T2& val){
        static_cast<std::gslice_array<T>&>(*this) -= std::valarray<T>((T)(val), size_);
        return *this;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    SubMatrixView<T> operator*=(const T2& val){
        static_cast<std::gslice_array<T>&>(*this) *= std::valarray<T>((T)(val), size_);
        return *this;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    SubMatrixView<T> operator/=(const T2& val){
        static_cast<std::gslice_array<T>&>(*this) /= std::valarray<T>((T)(val), size_);
        return *this;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    std::valarray<T> operator+(const T2& val){
        std::valarray<T> val_arr = *this;
        val_arr += (T)(val);
        return val_arr;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    std::valarray<T> operator-(const T2& val){
        std::valarray<T> val_arr = *this;
        val_arr -= (T)(val);
        return val_arr;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    std::valarray<T> operator*(const T2& val){
        std::valarray<T> val_arr = *this;
        val_arr *= (T)(val);
        return val_arr;
    }

    template<typename T2, std::enable_if_t<
        std::is_arithmetic_v<T2> && std::is_convertible_v<T, T2>, bool> = true>
    std::valarray<T> operator/(const T2& val){
        std::valarray<T> val_arr = *this;
        val_arr /= (T)(val);
        return val_arr;
    }
private:
    size_t size_;
    std::gslice g_;
};

} // namespace my_cnn
