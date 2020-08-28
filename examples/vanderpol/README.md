This example includes a modified `vanderpol.hpp` such that the Stan model holds a PETSc solver context as a member variable. This context is then passed to the function which calls the primal solve and adjoint solve.

Compile this example with (run from root cmdstan folder)

    STANCFLAGS=--allow-undefined USER_HEADER=examples/vanderpol/external_function_reverse_callback.hpp make examples/vanderpol/vanderpol

