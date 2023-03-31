#pragma once

#include <string>
#include <exception>

namespace my_cnn{

class MatrixSizeException : public std::exception{
public:
    MatrixSizeException(std::string msg = "") : msg_{msg} {}
    const char* what() const noexcept {return msg_.c_str();}
private:
    std::string msg_;
};

} // namespace my_cnn