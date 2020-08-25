#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/functor/adj_jac_apply.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err/check_nonzero_size.hpp>
#include <tuple>

#define PETSC_CLANGUAGE_CXX 1
#include <petsc.h>
#include <petscts.h>
#include <petscsys.h>

namespace stan {
namespace math {

namespace petsc {

template <class ExternalSolver>
class petsc_ts_functor {
    int N_;
    double* ic_mem_;  // Holds the input vector
    ExternalSolver solver_;

public:
    petsc_ts_functor() : N_(0), ic_mem_(nullptr), solver_(PETSC_COMM_WORLD) {}

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
        Vec petsc_ic = EigenVectorToPetscVecSeq(ic);

        // Initialize PETSc Vec to hold the results
        Vec petsc_out;
        PetscErrorCode ierr;
        ierr = VecDuplicate(petsc_ic, &petsc_out);CHKERRXX(ierr);

        ierr = VecCopy(petsc_ic, petsc_out);CHKERRXX(ierr);
        solver_.solve_forward(petsc_ic, petsc_out);

        // Convert PETSc Vec to Eigen
        Eigen::VectorXd out(N_);
        PetscVecToEigenVectorSeq(petsc_out, out);

        ierr = VecDestroy(&petsc_out);CHKERRXX(ierr);
        ierr = VecDestroy(&petsc_ic);CHKERRXX(ierr);

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
        PetscErrorCode ierr;
        Vec petsc_ic = EigenVectorToPetscVecSeq(ic);
        Vec petsc_adj = EigenVectorToPetscVecSeq(adj);

        // Calculate petsc_grad = adj * Jacobian(petsc_ic)
        ierr = VecCopy(petsc_ic, petsc_adj);CHKERRXX(ierr);

        solver_.solve_adjoint(petsc_ic, petsc_adj);

        // Convert PETSc Vec to Eigen
        Eigen::VectorXd out(N_);
        PetscVecToEigenVectorSeq(petsc_adj, out);

        ierr = VecDestroy(&petsc_adj);CHKERRXX(ierr);
        ierr = VecDestroy(&petsc_ic);CHKERRXX(ierr);

        return std::make_tuple(out);
    }
};
}  // namespace petsc

}  // namespace math
}  // namespace stan
