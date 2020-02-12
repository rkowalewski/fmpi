#include <fmpi/Dispatcher.hpp>

namespace fmpi {

std::ostream& operator<<(std::ostream& os, fmpi::Ticket const& ticket) {
  os << "{ id : " << ticket.id << " }";
  return os;
};
}
