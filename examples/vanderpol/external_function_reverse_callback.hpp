#include "petsc_vanderpol.hpp"

#include <stan/math/rev/functor/reverse_pass_callback.hpp>
#include <stan/math/rev/functor/arena_matrix.hpp>

#include <memory>

using namespace stan::math;

namespace vanderpol_model_namespace {

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, -1, 1>
my_vanderpol(const Eigen::Matrix<T0__, -1, 1>& initial_condition, std::ostream* pstream__) {
    throw std::logic_error("not implemented");  // this should never be called
}

template<>
Eigen::Matrix<double, -1, 1> my_vanderpol<double>(const Eigen::Matrix<double, -1, 1>& initial_condition, std::ostream* pstream__) {
    // auto solver_ = std::make_unique<PetscVanderpol>(PETSC_COMM_WORLD);
    PetscVanderpol solver(PETSC_COMM_WORLD);

    // std::cout << "double version is called" << std::endl;

    // Convert Eigen input to PETSc Vec
    Vec petsc_ic = petsc::EigenVectorToPetscVecSeq(value_of(initial_condition));

    // Initialize PETSc Vec to hold the results
    Vec petsc_out;
    PetscErrorCode ierr;
    ierr = VecDuplicate(petsc_ic, &petsc_out);CHKERRXX(ierr);
    solver.solve_forward(petsc_ic, petsc_out);

    // Convert PETSc Vec to Eigen
    int N_ = initial_condition.size();
    Eigen::VectorXd out(N_);
    petsc::PetscVecToEigenVectorSeq(petsc_out, out);

    ierr = VecDestroy(&petsc_out);CHKERRXX(ierr);
    ierr = VecDestroy(&petsc_ic);CHKERRXX(ierr);

    return out;
}

template<>
Eigen::Matrix<var, -1, 1> my_vanderpol<var>(const Eigen::Matrix<var, -1, 1>& initial_condition, std::ostream* pstream__) {
    auto solver_ = std::make_unique<PetscVanderpol>(PETSC_COMM_WORLD);

    // std::cout << "var version is called" << std::endl;

    // Convert Eigen input to PETSc Vec
    Vec petsc_ic = petsc::EigenVectorToPetscVecSeq(value_of(initial_condition));

    // Initialize PETSc Vec to hold the results
    Vec petsc_out;
    PetscErrorCode ierr;
    ierr = VecDuplicate(petsc_ic, &petsc_out);CHKERRXX(ierr);
    solver_->solve_forward(petsc_ic, petsc_out);

    // Convert PETSc Vec to Eigen
    int N_ = initial_condition.size();
    Eigen::VectorXd out(N_);
    petsc::PetscVecToEigenVectorSeq(petsc_out, out);

    arena_matrix<Eigen::VectorXd> res_val = out;
    arena_matrix<Eigen::Matrix<var, Eigen::Dynamic, 1>> res = res_val;
    arena_matrix<Eigen::Matrix<var, Eigen::Dynamic, 1>> ic_arena = initial_condition;

    reverse_pass_callback([=, solver_{std::move(solver_)}]() mutable {
        const auto& res_adj = to_ref(res.adj());
    
        // Convert Eigen input to PETSc Vec
        PetscErrorCode ierr;
        Vec petsc_ic = petsc::EigenVectorToPetscVecSeq(value_of(ic_arena));
        Vec petsc_adj = petsc::EigenVectorToPetscVecSeq(res_adj);

        // Calculate petsc_grad = adj * Jacobian(petsc_ic)
        solver_->solve_adjoint(petsc_ic, petsc_adj);

        // Convert PETSc Vec to Eigen
        Eigen::VectorXd out(N_);
        petsc::PetscVecToEigenVectorSeq(petsc_adj, out);

        ierr = VecDestroy(&petsc_adj);CHKERRXX(ierr);
        ierr = VecDestroy(&petsc_ic);CHKERRXX(ierr);

        ic_arena.adj() = out;
    });

    ierr = VecDestroy(&petsc_out);CHKERRXX(ierr);
    ierr = VecDestroy(&petsc_ic);CHKERRXX(ierr);

    return res;
}

} // Vanderpol_model_namespace
