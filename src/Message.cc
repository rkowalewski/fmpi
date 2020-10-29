#include <cstring>
#include <fmpi/Debug.hpp>
#include <fmpi/Message.hpp>
#include <sstream>

namespace fmpi {

int DefaultMessageHandler::operator()(
    message_type type, Message& message, MPI_Request& req) const {
  FMPI_ASSERT(
      type == message_type::ISEND || type == message_type::IRECV ||
      type == message_type::COPY);

  if (type == message_type::ISEND) {
    return send(message, req);
  }
  if (type == message_type::IRECV) {
    return recv(message, req);
  }

  FMPI_ASSERT(message.source() == message.dest());
  return lcopy(message);
}

int DefaultMessageHandler::send(const Message& message, MPI_Request& req) {
  std::ostringstream os;
  os << "Send : { dest : " << message.dest()
     << ", tag: " << message.sendtag() << "}";
  FMPI_DBG(os.str());
  return MPI_Isend(
      message.sendbuffer(),
      static_cast<int>(message.sendcount()),
      message.sendtype(),
      message.dest(),
      message.sendtag(),
      message.comm(),
      &req);
}
int DefaultMessageHandler::recv(Message& message, MPI_Request& req) {
#ifdef FMPI_DEBUG_ASSERT
  std::ostringstream os;
  os << "Receive : { source : " << message.source()
     << ", tag: " << message.recvtag() << "}";
  FMPI_DBG(os.str());
#endif
  return MPI_Irecv(
      message.recvbuffer(),
      static_cast<int>(message.recvcount()),
      message.recvtype(),
      message.source(),
      message.recvtag(),
      message.comm(),
      &req);
}
int DefaultMessageHandler::lcopy(Message& message) {
  FMPI_ASSERT(message.sendtype() == message.recvtype());
  MPI_Aint sendlb{};
  MPI_Aint sendextent{};

  FMPI_CHECK_MPI(
      MPI_Type_get_extent(message.sendtype(), &sendlb, &sendextent));

  std::memcpy(
      message.recvbuffer(),
      message.sendbuffer(),
      message.sendcount() * sendextent);

  return MPI_SUCCESS;
}
}  // namespace fmpi
