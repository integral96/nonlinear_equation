#pragma once

#include <memory>
#include <tuple>

#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/any.hpp>

namespace mpl = boost::mpl;

/*!
 * struct meta_loop
 */
template <size_t N, size_t I, class Closure>
typename boost::enable_if_t<(I == N)> is_meta_loop(Closure& closure) {}

template <size_t N, size_t I, class Closure>
typename boost::enable_if_t<(I < N)> is_meta_loop(Closure& closure) {
    closure.template apply<I>();
    is_meta_loop<N + 1, I + 2>(closure);
}
template <size_t N, class Closure>
void meta_loop(Closure& closure) {
    is_meta_loop<N, 0>(closure);
}


template<typename T>
struct accum_traits;
template<>
struct accum_traits<char> {
    typedef int type;
    static constexpr type zero() { return 0; }
};
template<>
struct accum_traits<short> {
    typedef int type;
    static constexpr type zero() { return 0; }
};
template<>
struct accum_traits<int> {
    typedef long type;
    static constexpr type zero() { return 0; }
};
template<>
struct accum_traits<unsigned int> {
    typedef unsigned long type;
    static constexpr type zero() { return 0; }
};
template<>
struct accum_traits<float> {
    typedef double type;
    static constexpr type zero() { return 0; }
};

///is_function
template<typename ...Args>
using typelist = mpl::vector_c<Args...>;

template<typename T>
struct is_function : boost::mpl::false_ {};

template<typename Func, typename ... Args>
struct is_function<Func(Args...)> : boost::mpl::true_ {
    typedef Func type;
    typedef typelist<Args...> param;
    static constexpr bool variadic = false;
};

template<typename Func, typename ... Args>
struct is_function<Func(Args..., ...)> : boost::mpl::true_ {
    typedef Func type;
    typedef typelist<Args...> param;
    static constexpr bool variadic = true;
};

///Matrix
template<size_t N, size_t M, typename T>
struct matrix_ ;
template<size_t N, size_t M, typename T>
std::ostream& operator << (std::ostream& os, const matrix_<N, M, T>& A);

template<size_t N, size_t M, typename T>
struct matrix_ {
    typedef boost::multi_array<T, 2> type;
    typedef T value_type;
    type MTRX{boost::extents[N][M]};

    constexpr T& operator () (size_t i, size_t j) {
        return MTRX[i][j];
    }
    constexpr T const& operator () (size_t i, size_t j) const {
        return MTRX[i][j];
    }
    constexpr T& at (size_t i, size_t j) {
        return MTRX[i][j];
    }
    matrix_<N, M, T>& operator = (matrix_<N, M, T> const& other) {
        MTRX = other.MTRX;
        return *this;
    }
    friend std::ostream& operator << (std::ostream& os, const matrix_<N, M, T>& A){
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                os << A.MTRX[i][j] << "\t";
            }
            os << std::endl;
        }
        return os;
    }
};

///math_expression
template <typename T>
struct math_base
{
    T& self() {
        return static_cast<T&>(*this);
    }
    const T& self() const {
        return static_cast<const T&>(*this);
    }
};
template <typename T> struct expression : math_base<T> {};

template <int N>
struct int_const : expression<int_const<N> > {
    static constexpr int value = N;
    typedef int_const<0> diff_type;
    diff_type diff() const {
        return diff_type();
    }
    template<typename T>
    size_t operator()(const T&) {
        return value;
    }
};

template<typename T>
struct is_int_constant : boost::mpl::false_ {};

template<int N>
struct is_int_constant<int_const<N>> : boost::mpl::true_ {};

template<typename T>
struct is_constant_value : boost::mpl::integral_c<int, 0> {};

template<int N>
struct is_constant_value<int_const<N>> : boost::mpl::integral_c<int, N> {};

///Scalar
template<typename T>
struct  scalar : expression<scalar<T>> {
    typedef T value_type;
    typedef int_const<0> diff_type;
    const value_type value;

    scalar(const value_type& value) : value(value) {}
    diff_type diff() const {
        return diff_type();
    }
    template<typename E>
    value_type operator() (const E& x) const {
        return value;
    }
};

template<typename E>
struct is_scalar : boost::mpl::false_ {};

template<typename T>
struct is_scalar<scalar<T>> : boost::mpl::true_ {};

//template <int N>
//struct variable : expression<variable<N> >
//{
//    template<int M>
//    struct diff_type {
//        typedef boost::mpl::if_c<(N == M), int_const<N>, mpl::na> type;
//    };
//    template<int M>
//    typename diff_type<M>::type diff() const {
//        return typename diff_type<M>::type();
//    }
//    template<typename T, int _size>
//    T operator()(const boost::array<T, _size> & vars) const {
//        return vars[N];
//    }
//};
template <size_t N>
struct variable : expression<variable<N> >
{
    typedef int_const<N> diff_type;

    diff_type diff() const {
        return diff_type();
    }
    template<typename T>
    T operator()(const T& x) const {
        return x;
    }
};
template <>
struct variable<0> : expression<variable<0> >
{
    typedef int_const<1> diff_type;
    diff_type diff() const {
        return diff_type();
    }
    template<typename T>
    T operator()(const T& x) const {
        return x;
    }
};


template<typename T>
scalar<T> _ (const T& val) {
    return scalar<T>(val);
}

template<typename E> struct negate_expression;

template<typename E>
struct negate_expression_type {
    typedef typename boost::mpl::if_c<is_int_constant<E>::value, int_const<- is_constant_value<E>::value>,
                            typename boost::mpl::if_c<is_scalar<E>::value, E, negate_expression<E>>::type>::type type;
};

template<typename T>
struct negate_expression : expression<negate_expression<T>> {
    typedef typename negate_expression_type<typename T::diff_type>::type diff_type;

    const T& e;
    negate_expression(const expression<T>& e) : e(e.self()) {}

    diff_type diff() const {
        return -e.diff();
    }

    template<typename E>
    E operator() (const E& x) const {
        return -e(x);
    }
};

template<typename T>
negate_expression<T> operator - (const expression<T>& e ) {
    return negate_expression<T>(e);
}
template<size_t N>
int_const<- N> operator - (const int_const<N>& ) {
    return int_const<- N>();
}
template<typename T>
scalar<T> operator - (const scalar<T>& e ) {
    return scalar<T>(-e.value);
}
template<typename T1, char OP, typename T2> struct operator_expression;
template<typename T1, char OP, typename T2>
using operator_type_plus_ = typename boost::mpl::if_c<is_int_constant<T1>::value || is_int_constant<T2>::value,
                                                int_const<is_constant_value<T1>::value + is_constant_value<T2>::value>,
                            typename boost::mpl::if_c<is_scalar<T1>::value || is_scalar<T2>::value,
                                                boost::is_same<T1, T2>, operator_expression<T1, OP, T2>>::type>::type;
template<typename T1, char OP, typename T2>
using operator_type_minus_ = typename boost::mpl::if_c<is_int_constant<T1>::value || is_int_constant<T2>::value,
                                                int_const<is_constant_value<T1>::value - is_constant_value<T2>::value>,
                            typename boost::mpl::if_c<is_scalar<T1>::value || is_scalar<T2>::value,
                                                boost::is_same<T1, T2>, operator_expression<T1, OP, T2>>::type>::type;

template<typename T1, char OP, typename T2>
struct operator_expression_type {
    typedef typename boost::mpl::if_c<OP == '+', operator_type_plus_<T1, OP, T2>,
                     boost::enable_if_<OP == '-', operator_type_minus_<T1, OP, T2>>>::type type;
};

template<char OP, typename T1, typename T2>
typename boost::enable_if_t<OP == '+', typename operator_expression_type<typename T1::diff_type, OP, typename T2::diff_type>::type>
operator_expression_diff(const T1& x1, const T2& x2) {
    return x1.diff() + x2.diff();
}
template<char OP, typename T1, typename T2>
typename boost::enable_if_t<OP == '-', typename operator_expression_type<typename T1::diff_type, OP, typename T2::diff_type>::type>
operator_expression_diff(const T1& x1, const T2& x2) {
    return x1.diff() - x2.diff();
}


template<typename T1, char OP, typename T2>
struct operator_expression : expression<operator_expression<T1, OP, T2>> {
    typedef typename operator_expression_type<typename T1::diff_type, OP, typename T2::diff_type>::type diff_type;

    const T1 x1;
    const T2 x2;

    operator_expression(const expression<T1>& x1, const expression<T2>& x2) : x1(x1.self()), x2(x2.self()) {}

    diff_type diff() const {
        return operator_expression_diff<OP>(x1, x2);
    }
    template<typename T>
    typename boost::enable_if_t<OP == '+', T> operator() (const T&x) const {
        return x1(x) + x2(x);
    }
    template<typename T>
    typename boost::enable_if_t<OP == '-', T> operator() (const T&x) const {
        return x1(x) - x2(x);
    }
};

template<typename T1, typename T2>
operator_expression<T1, '+', T2> operator + (const expression<T1>& x1, const expression<T2>& x2) {
    return operator_expression<T1, '+', T2>(x1, x2);
}
template<typename T>
const T& operator + (const expression<T>& x, const int_const<0>&) {
    return x.self();
}
template<typename T>
const T& operator + (const int_const<0>&, const expression<T>& x) {
    return x.self();
}

template<typename T1, typename T2>
operator_expression<T1, '-', T2> operator - (const expression<T1>& x1, const expression<T2>& x2) {
    return operator_expression<T1, '-', T2>(x1, x2);
}
template<typename T>
const T& operator - (const expression<T>& x, const int_const<0>&) {
    return x.self();
}
template<typename T>
const T& operator - (const int_const<0>&, const expression<T>& x) {
    return x.self();
}
template<size_t N1, size_t N2>
int_const<N1 + N2> operator + (const int_const<N1>&, const int_const<N2>&) {
    return int_const<N1 + N2>();
}
template<size_t N1>
int_const<N1> operator + (const int_const<N1>&, const int_const<0>&) {
    return int_const<N1>();
}
template<size_t N1>
int_const<N1> operator + (const int_const<0>&, const int_const<N1>&) {
    return int_const<N1>();
}
template<size_t N1, size_t N2>
int_const<N1 - N2> operator - (const int_const<N1>&, const int_const<N2>&) {
    return int_const<N1 - N2>();
}
template<size_t N1>
int_const<N1> operator - (const int_const<N1>&, const int_const<0>&) {
    return int_const<N1>();
}
template<size_t N1>
int_const<N1> operator - (const int_const<0>&, const int_const<N1>&) {
    return int_const<N1>();
}


template<class E, size_t I, size_t N>
struct jacobian1_closure {
    jacobian1_closure(const expression<E> &e, matrix_<N, N, boost::any> &result) : e(e.self()), result(result){}
    template<size_t J>
    void apply(){
        result(I, J) = e.template diff<J>();
    }
private:
    const E &e;
    matrix_<N, N, boost::any> &result;
};
template<class... E>
struct jacobian0_closure {
    static const size_t N = sizeof...(E);
    jacobian0_closure(const std::tuple<E...> &tp, matrix_<N, N, boost::any> &result) : tp(tp), result(result){}
    template<unsigned I>
    void apply(){
    typedef typename std::tuple_element<I, std::tuple<E...> >::type expr_type;
    jacobian1_closure<expr_type, I, N> closure(std::get<I>(tp), result);
        meta_loop<N>()(closure);
    }
private:
    const std::tuple<E...> &tp;
    matrix_<N, N, boost::any> &result;
};
template<class... E>
matrix_<sizeof...(E), sizeof...(E), boost::any> jacobian(const std::tuple<E...> &tp){
    constexpr size_t N = sizeof...(E);
    matrix_<N, N, boost::any> result;
    jacobian0_closure<E...> closure(tp, result);
    meta_loop<N>()(closure);
    return result;
}
template<typename T, class... E>
struct eval_tuple_closure {
    static constexpr std::size_t N = std::tuple_size<std::tuple<E...> >::value;
    eval_tuple_closure(const std::tuple<E...> &tp, const boost::array<T, N> &x, boost::array<T, N> &result) : tp(tp), x(x), result(result){}
    template<unsigned I>
    void apply(){
        result[I] = std::get<I>(tp)(x);
    }
private:
    const std::tuple<E...> &tp;
    const boost::array<T, N> &x;
    boost::array<T, N> &result;
};
template<typename T, class... E>
boost::array<T, sizeof...(E)> eval_tuple(const std::tuple<E...> &tp, const boost::array<T, sizeof...(E)> &x) {
    constexpr std::size_t N = sizeof...(E);
    boost::array<T, N> result;
    eval_tuple_closure<T, E...> closure(tp, x, result);
    meta_loop<N>(closure);
    return result;
}


template<typename T, typename ...Args>
boost::array<T, sizeof... (Args)> newton_solver(const std::tuple<Args...>& tp, const boost::array<T, sizeof... (Args)> x, size_t maxiter) {
    constexpr size_t N = sizeof... (Args);
    matrix_<N, N, boost::any> JACOBIAN = jacobian(tp);
    const double eps = 1e-12;
    size_t niter{};
    boost::array<T, N> fx;
    while (norm(fx = eval_tuple(tp, x)) > eps) {
        if(++niter > maxiter) throw std::runtime_error("Too mmany iteration");
        x -= solve(eval_jacobian<T, Args...>(JACOBIAN, x), fx);
    }
}
