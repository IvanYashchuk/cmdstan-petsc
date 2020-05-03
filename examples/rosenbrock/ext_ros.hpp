// template <typename T1, typename T2>
// typename boost::math::tools::promote_args<T1, T2>::type
// my_rosenbrock (const T1& A, const T2& B, std::ostream* pstream) {
//   typedef typename boost::math::tools::promote_args<T1, T2>::type T;

//   T C = A*B*B+A;

//   return C;
// }

#include <stan/math.hpp>
#include <stan/model/model_header.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/rev/core.hpp>

using namespace stan::math;

namespace rosenbrock_model_namespace {

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
my_rosenbrock(const Eigen::Matrix<T0__, -1, 1>& xy, std::ostream* pstream__) 
{
    // throw std::logic_error("not implemented");  // this should never be called
// typedef typename boost::math::tools::promote_args<T0__>::type T;
using T = stan::return_type_t<T0__>;
T x = xy(0);
T y = xy(1);
T res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
Eigen::Matrix<T, -1, 1> eres(1);
eres(0) = res;
return eres;
}

// template <typename T0__, typename T1__>
// typename boost::math::tools::promote_args<T0__, T1__>::type
// // stan::return_type_t<T0__, T1__>
// my_rosenbrock(const T0__& x,
//                   const T1__& y, std::ostream* pstream__) {
//                       throw std::logic_error("not implemented");  // this should never be called
//                     // typedef typename boost::math::tools::promote_args<T0__, T1__>::type T;
//                     // // using T = stan::return_type_t<T0__, T1__>;
//                     // T res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
//                     // return res;
//                   }

// template <typename T0__, typename T1__>
// T0__
// my_rosenbrock(const T1__& x,
//                   const T1__& y, std::ostream* pstream__) {
//                       throw std::logic_error("not implemented");  // this should never be called
//                   }

// template<>
// double my_rosenbrock<double, double>(const double& x, const double& y, std::ostream* pstream__) {
//     double res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
//     return res;
// }

// template<>
// var my_rosenbrock<var, var>(const var& x, const var& y, std::ostream* pstream__) {
//     double xv = x.val();
//     double yv = y.val();
//     double f_val = pow((1 - xv), 2) + (100 * pow((yv - pow(xv, 2)), 2));
//     double df_dx = -400 * xv * (yv - xv * xv) - 2 * (1 - xv);
//     double df_dy = 200 * (yv - xv * xv);
//     return var(new precomp_vv_vari(f_val, x.vi_, y.vi_, df_dx, df_dy));
// }

// template<>
// var my_rosenbrock<double, var>(const double& x, const var& y, std::ostream* pstream__) {
//     double xv = x;
//     double yv = y.val();
//     double f_val = pow((1 - xv), 2) + (100 * pow((yv - pow(xv, 2)), 2));
//     // double df_dx = -400 * xv * (yv - xv * xv) - 2 * (1 - xv);
//     double df_dy = 200 * (yv - xv * xv);
//     return var(new precomp_v_vari(f_val, y.vi_, df_dy));
// }

// template<>
// var my_rosenbrock<var, double>(const var& x, const double& y, std::ostream* pstream__) {
//     double xv = x.val();
//     double yv = y;
//     double f_val = pow((1 - xv), 2) + (100 * pow((yv - pow(xv, 2)), 2));
//     double df_dx = -400 * xv * (yv - xv * xv) - 2 * (1 - xv);
//     // double df_dy = 200 * (yv - xv * xv);
//     return var(new precomp_v_vari(f_val, x.vi_, df_dx));
// }

} // namespace
