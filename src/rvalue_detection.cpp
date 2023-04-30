#include <assert.h>
#include <iostream>
#include <type_traits>

struct S{
    S(): my_n(n++) {std::cout << "Constructed " << my_n << "\n";}
    ~S() {std::cout << "Destructor for " << my_n << '\n';}
    S(const S& s) : my_n(n++) {std::cout << "Copied " << s.my_n << " into " << my_n << "\n";}
    S(S&& s) : my_n(n++) {std::cout << "Moved " << s.my_n << " into " << my_n << "\n";}
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

template<typename T1, typename T2>
class Testing2{    
public:
    using T1_t = typename std::conditional_t<std::is_lvalue_reference_v<T1>, T1, typename std::remove_reference_t<T1>>;
    using T2_t = typename std::conditional_t<std::is_lvalue_reference_v<T2>, T2, typename std::remove_reference_t<T2>>;

    Testing2 (T1_t t1, T2_t t2) : t1_(std::forward<T1>(t1)), t2_(std::forward<T2>(t2)) {}

    T1_t t1_;
    T2_t t2_;
};

template<typename T1, typename T2>
auto makeTesting2(T1&& s1, T2&& s2){
    return Testing2<decltype(s1), decltype(s2)>(std::forward<T1>(s1), std::forward<T2>(s2));
}

void runRValueStorageTest(){
    S s;
    Testing<S&>{s}();

    Testing<S&&>{S{}}();
    std::cout << "Done\n";
}

void runCopyOrStorageTest(){
    const S s;
    auto T1 = makeTesting2(S{}, s);
    auto T2 = makeTesting2(makeTesting2(S{}, S{}), makeTesting2(s, s));
    T2.t2_.t1_();
    std::cout << "Done\n";
}

int main(){
    runRValueStorageTest();

    S::n = 0;
    std::cout << "-----------------------\n";

    runCopyOrStorageTest();
}