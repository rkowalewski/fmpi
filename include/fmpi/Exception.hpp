#ifndef FMPI_EXCEPTION_HPP
#define FMPI_EXCEPTION_HPP
#include <stdexcept>
namespace fmpi {
class NotImplemented : public std::logic_error {
 public:
  NotImplemented()
    : std::logic_error("not implemented"){};
};

class NotSupported : public std::logic_error {
 public:
  NotSupported()
    : std::logic_error("Unsupported Operation"){};
};
}  // namespace fmpi
#endif
