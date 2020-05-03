

#define PETSC_CLANGUAGE_CXX 1

#include "petsc_rosenbrock.hpp"

#include <petsc.h>
#include <petscts.h>
#include <petscsys.h>

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/functor/adj_jac_apply.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/err/check_nonzero_size.hpp>
#include <tuple>
#include <vector>

#include <iostream>

namespace stan {
namespace math {

namespace internal {

void PetscVecToEigen(const Vec& pvec, Eigen::VectorXd& evec)
{
   PetscScalar *pdata;
   // Returns a pointer to a contiguous array containing this processor's portion
   // of the vector data. For standard vectors this doesn't use any copies.
   // If the the petsc vector is not in a contiguous array then it will copy
   // it to a contiguous array.
   VecGetArray(pvec, &pdata);

   // Make the Eigen type a map to the data. Need to be mindful of anything that
   // changes the underlying data location like re-allocations.
   PetscInt size;
   VecGetSize(pvec, &size);
   evec = Eigen::Map<Eigen::VectorXd>(pdata, size);
   VecRestoreArray(pvec, &pdata);
}

inline Vec EigenToPetscVec(const Eigen::MatrixXd& evec)
{
  PetscErrorCode ierr;
  Vec pvec;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, evec.size(), evec.data(), &pvec);CHKERRXX(ierr);
  return pvec;
}

template <class ExternalSolver>
class petsc_op {
  int N_;
  double* x_mem_;  // Holds the input vector
  ExternalSolver solver_;

public:
  petsc_op() : N_(0), x_mem_(nullptr), solver_(PETSC_COMM_WORLD) {}

  /**
   * Call the PETSc function for the input vector
   *
   * @param x input vector.
   * @return Solution.
   */
  template <std::size_t size>
  Eigen::VectorXd operator()(const std::array<bool, size>& /* needs_adj */,
                             const Eigen::VectorXd& x) {
    // Save the input vector for multiply_adjoint_jacobian
    N_ = x.size();
    x_mem_ = ChainableStack::instance_->memalloc_.alloc_array<double>(N_);
    for (int n = 0; n < N_; ++n) {
      x_mem_[n] = x(n);
    }
    // std::cout << "HEHE" << x << std::endl;

    // Convert Eigen input to PETSc Vec
    // Vec petsc_x = EigenToPetscVec(x);
    Vec petsc_x;
    PetscErrorCode ierr;
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, x.size(), x.data(), &petsc_x);CHKERRXX(ierr);

    // Initialize PETSc Real to hold the results
    PetscReal petsc_out;

    solver_.solve_forward(petsc_x, &petsc_out);

    Eigen::VectorXd out(1);
    out(0) = petsc_out;

    return out;
  }

  /**
   * Compute the result of multiply the transpose of the adjoint vector times
   * the Jacobian of the softmax operator. It is more efficient to do this
   * without actually computing the Jacobian and doing the vector-matrix
   * product.
   *
   * @param adj Eigen::VectorXd of adjoints at the output of the softmax
   * @return Eigen::VectorXd of adjoints propagated through softmax operation
   */
  template <std::size_t size>
  std::tuple<Eigen::VectorXd> multiply_adjoint_jacobian(
      const std::array<bool, size>& /* needs_adj */,
      const Eigen::VectorXd& adj) const {

    Eigen::Map<vector_d> x(x_mem_, N_);

    Vec petsc_x, petsc_grad;
    PetscErrorCode ierr;
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, x.size(), x.data(), &petsc_x);CHKERRXX(ierr);
    ierr = VecDuplicate(petsc_x, &petsc_grad);

    solver_.solve_adjoint(petsc_x, petsc_grad, adj(0));

    Eigen::VectorXd out(N_);
    PetscVecToEigen(petsc_grad, out);

    // std::cout << "Gradient is " << out.transpose() << std::endl;

    return std::make_tuple(out);
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

using namespace stan::math;

namespace rosenbrock_model_namespace {

// template <typename T0__>
// Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
// my_rosenbrock(const Eigen::Matrix<T0__, -1, 1>& xy, std::ostream* pstream__) ;

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
my_rosenbrock(const Eigen::Matrix<T0__, -1, 1>& xy, std::ostream* pstream__) 
{
    throw std::logic_error("not implemented");  // this should never be called

// using T = stan::return_type_t<T0__>;
// T x = xy(0);
// T y = xy(1);
// T res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
// Eigen::Matrix<T, -1, 1> eres(1);
// eres(0) = res;
// return eres;
}

template<>
Eigen::Matrix<double, -1, 1> my_rosenbrock<double>(const Eigen::Matrix<double, -1, 1>& xy, std::ostream* pstream__) {
    matrix_v res = adj_jac_apply<internal::petsc_op<PetscRosenbrock>>(xy);
    return res.val();
    // throw std::logic_error("not implemented");  // this should never be called
    // double x = xy(0);
    // double y = xy(1);
    // double res = pow((1 - x), 2) + (100 * pow((y - pow(x, 2)), 2));
    // // Eigen::Matrix<double, 1, 1> eigen_res = {res};
    // Eigen::VectorXd eigen_res(1);
    // eigen_res(0) = res;
    // return eigen_res;
}

template<>
Eigen::Matrix<var, -1, 1> my_rosenbrock<var>(const Eigen::Matrix<var, -1, 1>& xy, std::ostream* pstream__) {
    // double xv = x.val();
    // double yv = y.val();
    // double f_val = pow((1 - xv), 2) + (100 * pow((yv - pow(xv, 2)), 2));
    // double df_dx = -400 * xv * (yv - xv * xv) - 2 * (1 - xv);
    // double df_dy = 200 * (yv - xv * xv);
    // return var(new precomp_vv_vari(f_val, x.vi_, y.vi_, df_dx, df_dy));

    return adj_jac_apply<internal::petsc_op<PetscRosenbrock>>(xy);
}

} // rosenbrock_model_namespace
