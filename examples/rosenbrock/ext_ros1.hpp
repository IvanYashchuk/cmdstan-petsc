#include "stan_petsc_interface.hpp"
#include "petsc_rosenbrock.hpp"

using namespace stan::math;

namespace rosenbrock_model_namespace {

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
my_rosenbrock(const Eigen::Matrix<T0__, -1, 1>& xy, std::ostream* pstream__) 
{
    throw std::logic_error("not implemented");  // this should never be called
}

template<>
Eigen::Matrix<double, -1, 1> my_rosenbrock<double>(const Eigen::Matrix<double, -1, 1>& xy, std::ostream* pstream__) {
    matrix_v res = adj_jac_apply<internal::petsc_functor<PetscRosenbrock>>(xy);
    return res.val();
}

template<>
Eigen::Matrix<var, -1, 1> my_rosenbrock<var>(const Eigen::Matrix<var, -1, 1>& xy, std::ostream* pstream__) {
    return adj_jac_apply<internal::petsc_functor<PetscRosenbrock>>(xy);
}

} // rosenbrock_model_namespace
