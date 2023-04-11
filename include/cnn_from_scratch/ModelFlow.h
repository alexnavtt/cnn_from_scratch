#pragma once

#include <vector>
#include <string_view>

namespace my_cnn{

enum ModelFlowMode{
    KERNEL,
    POOLING,
    FULLY_CONNECTED,
    OUTPUT
};
    
struct ModelFlow{
    std::vector<ModelFlowMode> stages;
    std::vector<size_t> indices;
    std::vector<std::string_view> names;
};

} // namespace my_cnn
