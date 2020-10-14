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
  COMMIT,
  BARRIER,
  WAITSOME,
};

#if 0
struct Envelope {
 private:
  mpi::Comm comm_{MPI_COMM_NULL};
  mpi::Rank peer_{MPI_PROC_NULL};
  mpi::Tag  tag_{};

 public:
  constexpr Envelope() = default;

  constexpr Envelope(mpi::Rank peer, mpi::Tag tag, mpi::Comm comm) noexcept
    : comm_(comm)
    , peer_(peer)
    , tag_(tag) {
  }

  [[nodiscard]] constexpr mpi::Rank peer() const noexcept {
    return peer_;
  }

  [[nodiscard]] constexpr mpi::Comm comm() const noexcept {
    return comm_;
  }

  [[nodiscard]] constexpr mpi::Tag tag() const noexcept {
    return tag_;
  }
};
#endif

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

#if 0
  constexpr Message(const Message&) = default;
  constexpr Message& operator=(Message const&) = default;

  constexpr Message(Message&&) noexcept = default;
  constexpr Message& operator=(Message&&) noexcept = default;
#endif

  constexpr const void* sendbuffer() const noexcept {
    return sendbuf_;
  }

  constexpr MPI_Datatype sendtype() const noexcept {
    return sendtype_;
  }

  constexpr std::size_t sendcount() const noexcept {
    return sendcount_;
  }

  constexpr int sendtag() const noexcept {
    return sendtag_;
  }

  constexpr void* recvbuffer() const noexcept {
    return recvbuf_;
  }

  constexpr MPI_Datatype recvtype() const noexcept {
    return recvtype_;
  }

  constexpr std::size_t recvcount() const noexcept {
    return recvcount_;
  }

  constexpr int recvtag() const noexcept {
    return recvtag_;
  }

  constexpr mpi::Comm comm() const noexcept {
    return comm_;
  }

  constexpr mpi::Rank source() const noexcept {
    return source_;
  }

  constexpr mpi::Rank dest() const noexcept {
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

}  // namespace fmpi

#endif
