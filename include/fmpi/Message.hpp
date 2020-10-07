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
  INVALID,  // DO NEVER USE
  COMMIT,
  BARRIER,
};

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

class Message {
 public:
  constexpr Message() = default;

  constexpr Message(
      void*        buf,
      std::size_t  count,
      MPI_Datatype type,
      mpi::Rank    peer,
      mpi::Tag     tag,
      mpi::Comm    comm) noexcept
    : buf_(buf)
    , count_(count)
    , mpi_type_(type)
    , envelope_(peer, tag, comm) {
  }

  constexpr Message(mpi::Rank peer, mpi::Tag tag, mpi::Comm comm) noexcept
    : envelope_(peer, tag, comm) {
  }

  template <class T>
  constexpr Message(
      gsl::span<T> span,
      mpi::Rank    peer,
      mpi::Tag     tag,
      mpi::Comm    comm) noexcept
    : Message(
          span.data(),
          span.size(),
          mpi::type_mapper<T>::type(),
          peer,
          tag,
          comm) {
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

  constexpr void* buffer() noexcept {
    return buf_;
  }

  [[nodiscard]] constexpr const void* buffer() const noexcept {
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

}  // namespace fmpi

#endif
