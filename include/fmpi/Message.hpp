#ifndef FMPI_MESSAGE_HPP
#define FMPI_MESSAGE_HPP

#include <cstddef>
#include <fmpi/Config.hpp>
#include <fmpi/mpi/Rank.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <gsl/span>
#include <variant>

namespace fmpi {

/// request type
enum class message_type : uint8_t
{
  IRECV = 0,
  ISEND,
  INVALID,  // DO NEVER USE
  ISENDRECV,
  COPY,
  WAITSOME,
  BARRIER,
  COMMIT,
  COMMIT_ALL,
};

class Message {
 public:
  constexpr Message() = default;

  constexpr Message(
      const void*  sendbuf,
      std::size_t  sendcount,
      MPI_Datatype sendtype,
      mpi::Rank    dst,
      int          sendtag,
      void*        recvbuf,
      std::size_t  recvcount,
      MPI_Datatype recvtype,
      mpi::Rank    source,
      int          recvtag,
      mpi::Comm    comm) noexcept
    : sendbuf_(sendbuf)
    , sendcount_(sendcount)
    , sendtype_(sendtype)
    , dst_(dst)
    , sendtag_(sendtag)
    , recvbuf_(recvbuf)
    , recvcount_(recvcount)
    , recvtype_(recvtype)
    , source_(source)
    , recvtag_(recvtag)
    , comm_(comm) {
  }

  template <class T, class U>
  constexpr Message(
      gsl::span<T> sendbuf,
      mpi::Rank    dst,
      int          sendtag,
      gsl::span<U> recvbuf,
      mpi::Rank    src,
      int          recvtag,
      mpi::Comm    comm) noexcept
    : Message(
          sendbuf.data(),
          sendbuf.size(),
          mpi::type_mapper<T>::type(),
          dst,
          sendtag,
          recvbuf.data(),
          recvbuf.size(),
          mpi::type_mapper<U>::type(),
          src,
          recvtag,
          comm) {
  }

  [[nodiscard]] constexpr const void* sendbuffer() const noexcept {
    return sendbuf_;
  }

  [[nodiscard]] constexpr MPI_Datatype sendtype() const noexcept {
    return sendtype_;
  }

  [[nodiscard]] constexpr std::size_t sendcount() const noexcept {
    return sendcount_;
  }

  [[nodiscard]] constexpr int sendtag() const noexcept {
    return sendtag_;
  }

  [[nodiscard]] constexpr void* recvbuffer() const noexcept {
    return recvbuf_;
  }

  [[nodiscard]] constexpr MPI_Datatype recvtype() const noexcept {
    return recvtype_;
  }

  [[nodiscard]] constexpr std::size_t recvcount() const noexcept {
    return recvcount_;
  }

  [[nodiscard]] constexpr int recvtag() const noexcept {
    return recvtag_;
  }

  [[nodiscard]] constexpr mpi::Comm comm() const noexcept {
    return comm_;
  }

  [[nodiscard]] constexpr mpi::Rank source() const noexcept {
    return source_;
  }

  [[nodiscard]] constexpr mpi::Rank dest() const noexcept {
    return dst_;
  }

  // Send buffer
  void const*  sendbuf_   = nullptr;
  std::size_t  sendcount_ = 0;
  MPI_Datatype sendtype_  = MPI_DATATYPE_NULL;
  mpi::Rank    dst_       = mpi::Rank::null();
  int          sendtag_   = 0;
  // Recv buffer
  void*        recvbuf_   = nullptr;
  std::size_t  recvcount_ = 0;
  MPI_Datatype recvtype_  = MPI_DATATYPE_NULL;
  mpi::Rank    source_    = mpi::Rank::null();
  int          recvtag_   = 0;
  // Comm
  MPI_Comm comm_ = MPI_COMM_NULL;
};

inline Message make_send(
    void const*  sendbuf,
    std::size_t  sendcount,
    MPI_Datatype sendtype,
    mpi::Rank    dst,
    int          sendtag,
    mpi::Comm    comm) noexcept {
  return Message(
      sendbuf,
      sendcount,
      sendtype,
      dst,
      sendtag,
      nullptr,
      0u,
      MPI_DATATYPE_NULL,
      mpi::Rank::null(),
      -1,
      comm);
}

inline Message make_receive(
    void*        recvbuf,
    std::size_t  recvcount,
    MPI_Datatype recvtype,
    mpi::Rank    source,
    int          recvtag,
    mpi::Comm    comm) noexcept {
  return Message(
      nullptr,
      0u,
      MPI_DATATYPE_NULL,
      mpi::Rank::null(),
      -1,
      recvbuf,
      recvcount,
      recvtype,
      source,
      recvtag,
      comm);
}

inline Message make_copy(
    void*        recvbuf,
    const void*  sendbuf,
    std::size_t  count,
    MPI_Datatype type) noexcept {
  return Message{
      sendbuf,
      count,
      type,
      mpi::Rank::null(),
      MPI_ANY_TAG,
      recvbuf,
      count,
      type,
      mpi::Rank::null(),
      MPI_ANY_TAG,
      MPI_COMM_NULL};
}

struct DefaultMessageHandler {
  int operator()(message_type type, Message& message, MPI_Request& req) const;

 private:
  static int send(const Message& message, MPI_Request& req);
  static int recv(Message& message, MPI_Request& req);
  static int lcopy(Message& message);

 public:
  static int sendrecv(Message& message, std::array<MPI_Request*, 2> reqs);
};

}  // namespace fmpi

#endif
