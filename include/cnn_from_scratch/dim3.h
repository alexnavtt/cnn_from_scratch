#pragma once

#include <ostream>

namespace my_cnn{
    
struct dim3{
    union{
        struct{
            unsigned x;
            unsigned y;
            unsigned z;
        };
        unsigned data[3] = {0, 0, 0};
    };

    friend std::ostream& operator << (std::ostream& os, dim3 dim){
        os << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
        return os;
    }
    bool operator==(const dim3& other) const{return (x == other.x) && (y == other.y) && (z == other.z);}
    bool operator!=(const dim3& other) const{return not (other == *this);}
};

} // namespace my_cnn
