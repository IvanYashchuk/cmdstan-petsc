#include <cmdstan/command.hpp>
#include <stan/services/error_codes.hpp>

#include <petsc.h>

int main(int argc, char* argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, 0, 0);CHKERRQ(ierr);

  try {
    ierr = cmdstan::command(argc, argv);CHKERRQ(ierr);
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    ierr = stan::services::error_codes::SOFTWARE;CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
