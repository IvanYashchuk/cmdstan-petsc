#ifndef STAN_PETSC_INTERFACE_HPP
#define STAN_PETSC_INTERFACE_HPP

#define PETSC_CLANGUAGE_CXX 1

#include <petsc.h>
#include <petscts.h>
#include <petscsys.h>

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/functor/adj_jac_apply.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/err/check_nonzero_size.hpp>
#include <tuple>

namespace stan {
namespace math {

namespace internal {

void PetscVecToEigen(const Vec& pvec, Eigen::VectorXd& evec)
{
    PetscErrorCode ierr;
    PetscScalar *pdata;
    // Returns a pointer to a contiguous array containing this processor's portion
    // of the vector data. For standard vectors this doesn't use any copies.
    // If the the petsc vector is not in a contiguous array then it will copy
    // it to a contiguous array.
    ierr = VecGetArray(pvec, &pdata);CHKERRXX(ierr);

    // Make the Eigen type a map to the data. Need to be mindful of anything that
    // changes the underlying data location like re-allocations.
    PetscInt size;
    ierr = VecGetSize(pvec, &size);CHKERRXX(ierr);
    evec = Eigen::Map<Eigen::VectorXd>(pdata, size);
    ierr = VecRestoreArray(pvec, &pdata);CHKERRXX(ierr);
}

template <class ExternalSolver>
class petsc_functor {
    int N_;
    double* ic_mem_;  // Holds the input vector
    ExternalSolver solver_;

public:
    petsc_functor() : N_(0), ic_mem_(nullptr), solver_(PETSC_COMM_WORLD) {}

    /**
     * Call the PETSc function for the input vector
     *
     * @param x input vector.
     * @return Solution.
     */
    template <std::size_t size>
    Eigen::VectorXd operator()(const std::array<bool, size>& /* needs_adj */,
                                const Eigen::VectorXd& ic) {
        // Save the input vector for multiply_adjoint_jacobian
        N_ = ic.size();
        ic_mem_ = ChainableStack::instance_->memalloc_.alloc_array<double>(N_);
        for (int n = 0; n < N_; ++n) {
            ic_mem_[n] = ic(n);
        }

        // Convert Eigen input to PETSc Vec
        Vec petsc_ic;
        PetscErrorCode ierr;
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, ic.size(), ic.data(), &petsc_ic);CHKERRXX(ierr);

        // Initialize PETSc Vec to hold the results
        Vec petsc_out;
        ierr = VecDuplicate(petsc_ic, &petsc_out);CHKERRXX(ierr);

        // petsc_out = forward_function(petsc_ic)
        solver_.solve_forward(petsc_ic, petsc_out);

        // Convert PETSc output to Eigen
        Eigen::VectorXd out(N_);
        PetscVecToEigen(petsc_out, out);

        return out;
    }

    /**
     * Compute the result of multiply the transpose of the adjoint vector times
     * the Jacobian of the PETSc forward function.
     *
     * @param adj Eigen::VectorXd of adjoints
     * @return Eigen::VectorXd adj*Jacobian
     */
    template <std::size_t size>
    std::tuple<Eigen::VectorXd> multiply_adjoint_jacobian(
        const std::array<bool, size>& /* needs_adj */,
        const Eigen::VectorXd& adj) const {

        // Restore input Eigen Vector
        Eigen::Map<vector_d> ic(ic_mem_, N_);

        // Convert Eigen input to PETSc Vec
        Vec petsc_ic, petsc_adj;
        PetscErrorCode ierr;
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, ic.size(), ic.data(), &petsc_ic);CHKERRXX(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, adj.size(), adj.data(), &petsc_adj);CHKERRXX(ierr);

        // Calculate petsc_grad = adj * Jacobian(petsc_ic)
        solver_.solve_adjoint(petsc_ic, petsc_adj);

        // Convert PETSc Vec to Eigen
        Eigen::VectorXd out(N_);
        PetscVecToEigen(petsc_adj, out);

        return std::make_tuple(out);
    }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
