#ifndef FMPI_MESSAGE_HPP
#define FMPI_MESSAGE_HPP

#include <cstddef>
#include <fmpi/mpi/Rank.hpp>
#include <fmpi/mpi/TypeMapper.hpp>
#include <gsl/span>

namespace fmpi {

/// request type
enum class message_type : uint8_t
{
  IRECV = 0,
  ISEND,

  INVALID
};

struct Envelope {
 private:
  mpi::Comm    comm_{MPI_COMM_NULL};
  mpi::Rank    peer_{};
  mpi::Tag     tag_{};

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

class Message {
 public:
  constexpr Message() = default;

  constexpr Message(
      mpi::Rank peer, mpi::Tag tag, mpi::Comm const& comm) noexcept
    : envelope_(peer, tag, comm) {
  }

  template <class T>
  constexpr Message(
      gsl::span<T>     span,
      mpi::Rank        peer,
      mpi::Tag         tag,
      mpi::Comm const& comm) noexcept
    : buf_(span.data())
    , count_(span.size())
    , mpi_type_(mpi::type_mapper<T>::type())
    , envelope_(peer, tag, comm)

  {
    set_buffer(span);
  }

  constexpr Message(const Message&) = default;
  constexpr Message& operator=(Message const&) = default;

  constexpr Message(Message&&) noexcept = default;
  constexpr Message& operator=(Message&&) noexcept = default;

  constexpr void set_buffer(void* buf, std::size_t count, MPI_Datatype type) {
    buf_      = buf;
    count_    = count;
    mpi_type_ = type;
  }

  template <class T>
  constexpr void set_buffer(gsl::span<T> buf) {
    set_buffer(buf.data(), buf.size(), mpi::type_mapper<T>::type());
  }

  constexpr void* writable_buffer() noexcept {
    return buf_;
  }

  [[nodiscard]] constexpr const void* readable_buffer() const noexcept {
    return buf_;
  }

  [[nodiscard]] constexpr MPI_Datatype type() const noexcept {
    return mpi_type_;
  }

  [[nodiscard]] constexpr std::size_t count() const noexcept {
    return count_;
  }

  [[nodiscard]] constexpr mpi::Rank peer() const noexcept {
    return envelope_.peer();
  }

  [[nodiscard]] constexpr mpi::Comm comm() const noexcept {
    return envelope_.comm();
  }

  [[nodiscard]] constexpr mpi::Tag tag() const noexcept {
    return envelope_.tag();
  }

 private:
  void*        buf_{};
  std::size_t  count_{};
  MPI_Datatype mpi_type_{};
  Envelope     envelope_{};
};

//static_assert(sizeof(Envelope) == 12);
//static_assert(sizeof(MPI_Datatype) == 8);
//static_assert(alignof(Message) == 8);
//static_assert(sizeof(Message) == 32);

class NonblockingMessageHandler {};

}  // namespace fmpi

#endif
