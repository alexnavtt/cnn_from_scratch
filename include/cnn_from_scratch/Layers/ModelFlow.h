#pragma once

#include <vector>
#include <string>

namespace my_cnn{

enum ModelFlowMode{
    UNSPECIFIED,
    KERNEL,
    POOLING,
    FULLY_CONNECTED,
    ACTIVATION,
    OUTPUT
};

} // namespace my_cnn
