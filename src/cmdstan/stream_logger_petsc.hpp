#ifndef CMDSTAN_CALLBACKS_STREAM_LOGGER_PETSC_HPP
#define CMDSTAN_CALLBACKS_STREAM_LOGGER_PETSC_HPP

#include <stan/callbacks/logger.hpp>
#include <ostream>
#include <string>
#include <sstream>

#define PETSC_CLANGUAGE_CXX 1
#include <petsc.h>

namespace stan {
namespace callbacks {

/**
 * <code>stream_logger</code> is an implementation of
 * <code>logger</code> that writes messages to separate
 * std::stringstream outputs.
 */
class stream_logger_petsc : public logger {
 private:
  std::ostream& debug_;
  std::ostream& info_;
  std::ostream& warn_;
  std::ostream& error_;
  std::ostream& fatal_;
  PetscMPIInt rank_;

 public:
  /**
   * Constructs a <code>stream_logger</code> with an output
   * stream for each log level.
   *
   * @param[in,out] debug stream to output debug messages
   * @param[in,out] info stream to output info messages
   * @param[in,out] warn stream to output warn messages
   * @param[in,out] error stream to output error messages
   * @param[in,out] fatal stream to output fatal messages
   */
  stream_logger_petsc(std::ostream& debug, std::ostream& info, std::ostream& warn,
                std::ostream& error, std::ostream& fatal)
      : debug_(debug), info_(info), warn_(warn), error_(error), fatal_(fatal) {
        PetscMPIInt mpi_error = MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);CHKERRXX(mpi_error);
      }

  void debug(const std::string& message) {
    if (rank_ == 0)
      debug_ << message << std::endl;
  }

  void debug(const std::stringstream& message) {
    if (rank_ == 0)
      debug_ << message.str() << std::endl;
  }

  void info(const std::string& message) {
    if (rank_ == 0)
      info_ << message << std::endl;
  }

  void info(const std::stringstream& message) {
    if (rank_ == 0)
      info_ << message.str() << std::endl;
  }

  void warn(const std::string& message) {
    if (rank_ == 0)
      warn_ << message << std::endl;
  }

  void warn(const std::stringstream& message) {
    if (rank_ == 0)
      warn_ << message.str() << std::endl;
  }

  void error(const std::string& message) {
    if (rank_ == 0)
      error_ << message << std::endl;
  }

  void error(const std::stringstream& message) {
    if (rank_ == 0)
      error_ << message.str() << std::endl;
  }

  void fatal(const std::string& message) {
    if (rank_ == 0)
      fatal_ << message << std::endl;
  }

  void fatal(const std::stringstream& message) {
    if (rank_ == 0)
      fatal_ << message.str() << std::endl;
  }
};

}  // namespace callbacks
}  // namespace stan
#endif
