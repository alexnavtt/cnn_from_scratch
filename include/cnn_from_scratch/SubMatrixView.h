#pragma once

#include <valarray>

namespace my_cnn{
    
template <typename T>
class SubMatrixView : public std::gslice_array<T>{
public:
    using std::gslice_array<T>::operator=;
    SubMatrixView(std::valarray<T>& arr, const std::gslice& g) :
    std::gslice_array<T>(arr[g]),
    g_(g)
    {}

    operator std::valarray<T>(){
        return std::valarray<T>(*dynamic_cast<std::gslice_array<T>*>(this));
    }
private:
    std::gslice g_;
};

} // namespace my_cnn
