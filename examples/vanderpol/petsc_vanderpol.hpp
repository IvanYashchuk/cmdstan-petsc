#ifndef PETSC_VANDERPOL_HPP
#define PETSC_VANDERPOL_HPP

#define PETSC_CLANGUAGE_CXX 1

/*
This example demonstrates use of PETSc with Stan on a single processor.
We consider the van der Pol equation from PETSc /ts/tutorials/ex20opt_ic.c

This code demonstrates how to solve an ODE-constrained optimization problem with TAO, TSAdjoint and TS.
The nonlinear problem is written in an ODE equivalent form.
The gradient is computed with the discrete adjoint of an implicit method or an explicit method, see ex20adj.c for details.
*/

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  TS        ts;
  PetscReal mu;
  PetscReal next_output;
  TSTrajectory tj;

  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                       /* Jacobian matrix for ODE */
  Mat       Jacp;                    /* JacobianP matrix for ODE*/
  Mat       H;                       /* Hessian matrix for optimization */
  Vec       U,Lambda[1],Mup[1];      /* first-order adjoint variables */
  Vec       Lambda2[2];              /* second-order adjoint variables */
  Vec       Ihp1[1];                 /* working space for Hessian evaluations */
  Vec       Dir;                     /* direction vector */
  PetscReal ob[2];                   /* observation used by the cost function */
  PetscBool implicitform;            /* implicit ODE? */
};

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRXX(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRXX(ierr);
  ierr = VecGetArray(F,&f);CHKERRXX(ierr);
  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu*((1.0-u[0]*u[0])*u[1] - u[0]) ;
  ierr = VecRestoreArrayRead(U,&u);CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRXX(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRXX(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat B, void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(U,&u);CHKERRXX(ierr);
  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = user->mu*(1.0 + 2.0*u[0]*u[1]);   J[1][1] = a - user->mu*(1.0-u[0]*u[0]);
  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRXX(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRXX(ierr);
  ierr    = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr    = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  if (A != B) {
    ierr  = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr  = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
// static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
// {
//   PetscErrorCode    ierr;
//   const PetscScalar *u;
//   PetscReal         tfinal, dt;
//   User              user = (User)ctx;
//   Vec               interpolatedU;

//   PetscFunctionBeginUser;
//   ierr = TSGetTimeStep(ts,&dt);CHKERRXX(ierr);
//   ierr = TSGetMaxTime(ts,&tfinal);CHKERRXX(ierr);

//   while (user->next_output <= t && user->next_output <= tfinal) {
//     ierr = VecDuplicate(U,&interpolatedU);CHKERRXX(ierr);
//     ierr = TSInterpolate(ts,user->next_output,interpolatedU);CHKERRXX(ierr);
//     ierr = VecGetArrayRead(interpolatedU,&u);CHKERRXX(ierr);
//     ierr = PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
//                        (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(u[0]),
//                        (double)PetscRealPart(u[1]));CHKERRXX(ierr);
//     ierr = VecRestoreArrayRead(interpolatedU,&u);CHKERRXX(ierr);
//     ierr = VecDestroy(&interpolatedU);CHKERRXX(ierr);
//     user->next_output += 0.1;
//   }
//   PetscFunctionReturn(0);
// }

class PetscVanderpol
{
public:
  /// User-defined application context
  struct _n_User user;

  explicit PetscVanderpol(MPI_Comm comm) : user()
  {
    // std::cout << "Constructor is called!" << std::endl;
    // PetscBool monitor = PETSC_TRUE;
    PetscErrorCode ierr;

    /* Set runtime options */
    user.next_output  = 0.0;
    user.mu           = 1.0e3;
    user.steps        = 0;
    user.ftime        = 0.5;
    user.implicitform = PETSC_TRUE;

    /* Create necessary matrix and vectors, solve same ODE on every process */
    ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRXX(ierr);
    ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRXX(ierr);
    ierr = MatSetFromOptions(user.A);CHKERRXX(ierr);
    ierr = MatSetUp(user.A);CHKERRXX(ierr);
    ierr = MatCreateVecs(user.A,&user.U,NULL);CHKERRXX(ierr);
    ierr = MatCreateVecs(user.A,&user.Dir,NULL);CHKERRXX(ierr);
    ierr = MatCreateVecs(user.A,&user.Lambda[0],NULL);CHKERRXX(ierr);
    ierr = MatCreateVecs(user.A,&user.Lambda2[0],NULL);CHKERRXX(ierr);
    ierr = MatCreateVecs(user.A,&user.Ihp1[0],NULL);CHKERRXX(ierr);

    /* Create timestepping solver context */
    ierr = TSCreate(PETSC_COMM_WORLD,&user.ts);CHKERRXX(ierr);
    ierr = TSSetEquationType(user.ts,TS_EQ_ODE_EXPLICIT);CHKERRXX(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */

    // if (user.implicitform)
    ierr = TSSetIFunction(user.ts,NULL,IFunction,&user);CHKERRXX(ierr);
    ierr = TSSetIJacobian(user.ts,user.A,user.A,IJacobian,&user);CHKERRXX(ierr);
    ierr = TSSetType(user.ts,TSCN);CHKERRXX(ierr);

    ierr = TSSetMaxTime(user.ts,user.ftime);CHKERRXX(ierr);
    ierr = TSSetExactFinalTime(user.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRXX(ierr);

    // if (monitor) {
    //   ierr = TSMonitorSet(user.ts,Monitor,&user,NULL);CHKERRXX(ierr);
    // }

    /* Set runtime options */
    ierr = TSSetFromOptions(user.ts);CHKERRXX(ierr);
  
    /* Save trajectory of solution so that TSAdjointSolve() may be used */
    ierr = TSSetSaveTrajectory(user.ts);CHKERRXX(ierr);

    ierr = TSSetMaxSNESFailures(user.ts, 50);CHKERRXX(ierr);

    // Make TS save the trajectory in memory (by default is to file)
    ierr = TSGetTrajectory(user.ts, &user.tj);CHKERRXX(ierr);
    ierr = TSTrajectorySetType(user.tj, user.ts, TSTRAJECTORYMEMORY);
  }

  virtual ~PetscVanderpol()
  {
    PetscErrorCode ierr;
    // std::cout << "Desctructor is called!" << std::endl;
    /* Free work space.  All PETSc objects should be destroyed when they are no longer needed. */
    ierr = MatDestroy(&user.A);CHKERRXX(ierr);
    ierr = VecDestroy(&user.U);CHKERRXX(ierr);
    ierr = VecDestroy(&user.Lambda[0]);CHKERRXX(ierr);
    ierr = VecDestroy(&user.Lambda2[0]);CHKERRXX(ierr);
    ierr = VecDestroy(&user.Ihp1[0]);CHKERRXX(ierr);
    ierr = VecDestroy(&user.Dir);CHKERRXX(ierr);
    ierr = TSDestroy(&user.ts);CHKERRXX(ierr);
  }

  /// Solve the forward problem
  PetscErrorCode solve_forward(Vec initial_condition, Vec out) const
  {
    // std::cout << "solve_forward is called!" << std::endl;
    PetscErrorCode ierr;
    ierr = VecCopy(initial_condition, out);CHKERRXX(ierr);
    ierr = TSSolve(user.ts, out);CHKERRXX(ierr);
    PetscFunctionReturn(0);
  }

  /// Calculate the gradient (based on FormFunctionGradient)
  PetscErrorCode solve_adjoint(Vec initial_condition, Vec grad) const
  {
    // std::cout << "solve_adjoint is called!" << std::endl;
    PetscErrorCode    ierr;

    PetscFunctionBeginUser;
    ierr = VecCopy(initial_condition, user.U);CHKERRXX(ierr);

    ierr = TSSetTime(user.ts, 0.0);CHKERRXX(ierr);
    ierr = TSSetStepNumber(user.ts, 0);CHKERRXX(ierr);
    ierr = TSResetTrajectory(user.ts);CHKERRXX(ierr);
    ierr = TSSetTimeStep(user.ts, 0.001);CHKERRXX(ierr); /* can be overwritten by command line options */
    ierr = TSSetFromOptions(user.ts);CHKERRXX(ierr);

    ierr = TSSolve(user.ts, user.U);CHKERRXX(ierr);

    ierr = TSSetCostGradients(user.ts, 1, &grad, NULL);CHKERRXX(ierr);
    ierr = TSAdjointSolve(user.ts);CHKERRXX(ierr);
    ierr = TSResetTrajectory(user.ts);CHKERRXX(ierr);
    PetscFunctionReturn(0);
  }
};

#endif
