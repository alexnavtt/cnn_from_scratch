#pragma once

#include <ostream>

namespace my_cnn{

struct dim2{
    union{
        struct{
            unsigned x;
            unsigned y;
        };
        unsigned data[2] = {0, 0};
    };

    dim2() = default;
    dim2(unsigned val) : x(val), y(val) {}
    dim2(unsigned x_, unsigned y_) : x(x_), y(y_) {}

    friend std::ostream& operator << (std::ostream& os, dim3 dim){
        os << "(" << dim.x << ", " << dim.y << ")";
        return os;
    }
    bool operator==(const dim2& other) const{return (x == other.x) && (y == other.y);}
    bool operator!=(const dim2& other) const{return not (other == *this);}
};
    
struct dim3{
    union{
        struct{
            unsigned x;
            unsigned y;
            unsigned z;
        };
        unsigned data[3] = {0, 0, 0};
    };

    dim3() = default;
    dim3(unsigned val) : x(val), y(val), z(val) {}
    dim3(unsigned x_, unsigned y_, unsigned z_) : x(x_), y(y_), z(z_) {}

    friend std::ostream& operator << (std::ostream& os, dim3 dim){
        os << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
        return os;
    }
    bool operator==(const dim3& other) const{return (x == other.x) && (y == other.y) && (z == other.z);}
    bool operator!=(const dim3& other) const{return not (other == *this);}
    dim3 operator+(const dim3& other) const{return dim3(x + other.x, y + other.y, z + other.z);}
    dim3 slice() const noexcept {return {x, y, 1};}
};

} // namespace my_cnn
