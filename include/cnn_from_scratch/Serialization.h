#pragma once

#include <string>
#include <limits>
#include <istream>
#include <type_traits>

namespace my_cnn{

namespace serialization{

/**
 * Given an input stream, ignore everything until the next line 
 */
static inline void clearLine(std::istream& S){
    S.ignore(std::numeric_limits<int>::max(), '\n');
}

/**
 * Given an output stream, place a label and value on it's own line
 * such that that output is of the form "label value". This function
 * acts as the inverse of serialization::expect
 */
template<typename T>
static inline void place(std::ostream& os, const T& val, const std::string& label){
    os << label << ' ' << val << '\n'; 
}

/**
 * Given an input stream, check that the next entry in the stream is as expected.
 * If it is indeed the expected string, get the value immediately proceeding that
 * string and return it as the specified type. This function acts as the inverse 
 * of serialization::place
 * @param is        The input stream
 * @param expected  The expected string in the stream
 * @return          The T value immediately after expected in the stream
 */  
template<typename T>
static inline T expect(std::istream& is, const std::string& expected){
    std::string result(expected.size(), ' ');
    is.read(result.data(), result.size());
    if (expected != result){
        throw std::runtime_error("Error loading Kernel. Expected value was \"" + expected + "\" but \"" + result + "\" was read instead");
    }

    if constexpr (not std::is_void_v<T>){
        T output;
        is >> output;
        clearLine(is);
        return output;
    }else{
        clearLine(is);
    }
}

} // namespace serialization

} // namespace my_cnn
