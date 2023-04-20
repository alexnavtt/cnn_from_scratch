#include <iostream>
#include <type_traits>

struct S{
    S(){std::cout << "Constructed " << n++ << "\n";}
    void operator()() const && {std::cout << "S rvalue reference from " << my_n << "\n";}
    void operator()() const & {std::cout << "S lvalue reference from " << my_n << "\n";}
    static int n;
    int my_n = n;
};

int S::n = 0;

template<typename T>
class Testing{
public:

    template<typename = std::enable_if_t<std::is_reference_v<T>>>
    Testing(T s) : s_(&s) {}

    void operator()() const && {
        std::forward<T>(*s_)();
    }

    S* s_;
};

template<typename T>
auto makeTesting(T&& s){
    return Testing<decltype(std::forward<T>(s))>(std::forward<T>(s));
}

int main(){

    S s;
    Testing<S&>{s}();

    Testing<S&&>{S{}}();

    makeTesting(S{})();

    makeTesting(s)();
}