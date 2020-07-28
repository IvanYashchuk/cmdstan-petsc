
# STAN_MPI=false
# CXX=mpicxx
# TBB_CXX_TYPE=clang

ifndef PETSC_DIR
$(error PETSC_DIR is not set. Please set PETSC_DIR via export PETSC_DIR=/path/to/petsc)
endif

include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules
#include ${PETSC_DIR}/lib/petsc/conf/test

CXXFLAGS += -I$(PETSC_DIR)/include
LDLIBS += $(PETSC_SYS_LIB)
