#pragma once

namespace my_cnn{

using uint = unsigned;

enum PoolingType{
    MAX,
    MIN,
    AVG
};
    
struct Pooling{
    uint dim1   = 1;
    uint dim2   = 1;
    uint stride = 1;
    PoolingType type = PoolingType::MAX;
};

} // namespace my_cnn
