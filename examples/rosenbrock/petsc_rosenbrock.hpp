#ifndef PETSC_ROSENBROCK_HPP
#define PETSC_ROSENBROCK_HPP

#define PETSC_CLANGUAGE_CXX 1

/*
This example demonstrates use of the TAO package to
solve an unconstrained minimization problem on a single processor.
We minimize the extended Rosenbrock function:
sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 )
*/

#include <petsc.h>

#include <iostream>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/
typedef struct {
  PetscInt  n;          /* dimension */
  PetscReal alpha;   /* condition parameter */
} AppCtx;

class PetscRosenbrock
{
public:
  /// User-defined application context
  AppCtx appctx_;

  explicit PetscRosenbrock(MPI_Comm comm) : appctx_()
  {
    /* Initialize problem parameters */
    appctx_.n = 2;
    appctx_.alpha = 100.0;

    /* Check for command line arguments to override defaults */
    PetscErrorCode ierr;
    PetscBool flg;
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &appctx_.n, &flg);CHKERRXX(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, "-alpha", &appctx_.alpha, &flg);CHKERRXX(ierr);
  }

  virtual ~PetscRosenbrock() {}

  /// Solve the forward problem
  PetscErrorCode solve_forward(Vec X, PetscReal *f) const
  {
    PetscInt i, nn = appctx_.n/2;
    PetscErrorCode ierr;
    PetscReal ff = 0;
    PetscReal alpha=appctx_.alpha;
    PetscReal t1, t2;
    PetscReal *x;

    /* Get pointers to vector data */
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    // PetscPrintf(PETSC_COMM_SELF,"Forward input:\n");
    // PetscRealView(2, x, PETSC_VIEWER_STDOUT_SELF);

    /* Compute G(X) */
    for (i=0; i<nn; i++){
        t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
        ff += alpha*t1*t1 + t2*t2;
    }

    /* Restore vectors */
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    *f=ff;
    PetscFunctionReturn(0);
  }

  /// Calculate the gradient
  PetscErrorCode solve_adjoint(Vec X, Vec G, const PetscReal& adj) const
  {
    PetscInt i, nn = appctx_.n/2;
    PetscErrorCode ierr;
    PetscReal alpha=appctx_.alpha;
    PetscReal t1, t2;
    PetscReal *x, *g;

    /* Get pointers to vector data */
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    ierr = VecGetArray(G, &g);CHKERRQ(ierr);

    // PetscPrintf(PETSC_COMM_SELF,"Backward input:\n");
    // PetscRealView(2, x, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "PETSCREALADJ:  " << adj << std::endl;
    // PetscReal ff = 0;
  
    /* Compute G(X) */
    for (i=0; i<nn; i++){
        t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
        // ff += alpha*t1*t1 + t2*t2;
        g[2*i] = adj*(-4*alpha*t1*x[2*i]-2.0*t2);
        g[2*i+1] = adj*(2*alpha*t1);
    }

    // PetscPrintf(PETSC_COMM_SELF,"PETSc objective: \n");
    // std::cout << ff << std::endl;

    // PetscPrintf(PETSC_COMM_SELF,"PETSc gradient: \n");
    // PetscRealView(2, g, PETSC_VIEWER_STDOUT_SELF);

    /* Restore vectors */
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
};

#endif