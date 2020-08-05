// #include <stan/math.hpp>
// #include <boost/math/tools/promotion.hpp>

// using namespace stan::math;

namespace rosenbrock_model_namespace {

template <typename T0__>
Eigen::Matrix<stan::promote_args_t<T0__>, -1, 1>
my_rosenbrock(const Eigen::Matrix<T0__, -1, 1>& xy, std::ostream* pstream__)
{
    // throw std::logic_error("not implemented");  // this should never be called
    // typedef typename boost::math::tools::promote_args<T0__>::type T;
    // if (xy.size() > 2) {
    //     throw std::logic_error("This function is implemented only for input of size 2.");
    // }
    using T = stan::return_type_t<T0__>;
    T x = xy(0);
    T y = xy(1);
    T res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
    Eigen::Matrix<T, -1, 1> out(1);
    out(0) = res;
    return out;
}

} // rosenbrock_model_namespace
