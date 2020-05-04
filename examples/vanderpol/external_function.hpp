#include "stan_petsc_interface.hpp"
#include "petsc_vanderpol.hpp"

using namespace stan::math;

namespace vanderpol_model_namespace {

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
my_vanderpol(const Eigen::Matrix<T0__, -1, 1>& initial_condition, std::ostream* pstream__) 
{
    throw std::logic_error("not implemented");  // this should never be called
}

template<>
Eigen::Matrix<double, -1, 1> my_vanderpol<double>(const Eigen::Matrix<double, -1, 1>& initial_condition, std::ostream* pstream__) {
    matrix_v res = adj_jac_apply<internal::petsc_functor<PetscVanderpol>>(initial_condition);
    return res.val();
}

template<>
Eigen::Matrix<var, -1, 1> my_vanderpol<var>(const Eigen::Matrix<var, -1, 1>& initial_condition, std::ostream* pstream__) {
    return adj_jac_apply<internal::petsc_functor<PetscVanderpol>>(initial_condition);
}

} // Vanderpol_model_namespace
