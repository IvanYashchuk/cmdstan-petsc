
# STAN_MPI=false
# CXX=mpicxx
# TBB_CXX_TYPE=clang

ifndef PETSC_DIR
$(error PETSC_DIR is not set. Please set PETSC_DIR via export PETSC_DIR=/path/to/petsc)
endif

ifdef STAN_MPI
$(error PETSc cannot be used together with Stan MPI implementation)
endif

ifdef STAN_NUM_THREADS
ifneq ($(STAN_NUM_THREADS),1)
$(warning "MPI Stan-PETSc programs use process-based parallelism and run duplicate computations on Stan side, threads might hurt performance. Consider setting STAN_NUM_THREADS to 1.")
endif
endif

include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/lib/petsc/conf/test

CXXFLAGS += -I$(PETSC_DIR)/include
LDLIBS += $(PETSC_SYS_LIB)
