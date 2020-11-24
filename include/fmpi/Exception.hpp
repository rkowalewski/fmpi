#ifndef FMPI_EXCEPTION_HPP
#define FMPI_EXCEPTION_HPP
#include <stdexcept>
namespace fmpi {
class NotImplementedException : public std::logic_error {
 public:
  NotImplementedException()
    : std::logic_error("not implemented"){};
};

class NotSupportedException : public std::logic_error {
 public:
  NotSupportedException()
    : std::logic_error("Unsupported Operation"){};
};
}  // namespace fmpi
#endif
