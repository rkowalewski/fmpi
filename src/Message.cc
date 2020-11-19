#include <cstring>
#include <fmpi/Debug.hpp>
#include <fmpi/Message.hpp>
#include <sstream>

namespace fmpi {

namespace detail {

static constexpr std::size_t eager_limit = 128;

static inline std::size_t message_length(
    MPI_Datatype type, std::size_t count) {
  MPI_Aint lb{};
  MPI_Aint extent{};

  FMPI_CHECK_MPI(MPI_Type_get_extent(type, &lb, &extent));

  return count * extent;
}
}  // namespace detail

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
#if FMPI_DEBUG_ASSERT
  std::ostringstream os;
  os << "Send : { dest : " << message.dest() << ", tag: " << message.sendtag()
     << "}";
  FMPI_DBG(os.str());
#endif

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
#if FMPI_DEBUG_ASSERT
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

  std::memcpy(
      message.recvbuffer(),
      message.sendbuffer(),
      detail::message_length(message.sendtype(), message.sendcount()));

  return MPI_SUCCESS;
}

int DefaultMessageHandler::sendrecv(
    Message& message, std::array<MPI_Request*, 2> reqs, bool blocking) {
  if (blocking) {
#if FMPI_DEBUG_ASSERT
    {
      std::ostringstream os;
      os << "SendRecv : { source : " << message.source();
      os << ", recvtag: " << message.recvtag();
      os << ", dest: " << message.dest();
      os << ", sendtag: " << message.sendtag();
      os << "}";
      FMPI_DBG(os.str());
    }
#endif

    return MPI_Sendrecv(
        message.sendbuffer(),
        static_cast<int>(message.sendcount()),
        message.sendtype(),
        message.dest(),
        message.sendtag(),
        message.recvbuffer(),
        static_cast<int>(message.recvcount()),
        message.recvtype(),
        message.source(),
        message.recvtag(),
        message.comm(),
        MPI_STATUS_IGNORE);
  } else {
    FMPI_CHECK_MPI(recv(message, *reqs[1]));
    return send(message, *reqs[0]);
  }
}
}  // namespace fmpi
