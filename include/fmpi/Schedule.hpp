#ifndef FMPI_SCHEDULE_HPP
#define FMPI_SCHEDULE_HPP

#include <fmpi/Config.hpp>
#include <fmpi/Math.hpp>
#include <fmpi/detail/Assert.hpp>
#include <fmpi/mpi/Environment.hpp>

namespace fmpi {

using namespace mpi;

class FlatHandshake {
 public:
  static constexpr auto NAME = std::string_view("Ring");

  constexpr FlatHandshake() = default;

  constexpr FlatHandshake(Context const& ctx) FMPI_NOEXCEPT
    : FlatHandshake(ctx.size(), ctx.rank()) {
  }

  constexpr FlatHandshake(uint32_t nodes, Rank rank) FMPI_NOEXCEPT
    : rank_(rank),
      nodes_(nodes) {
  }

  constexpr Rank sendRank(uint32_t phase) const FMPI_NOEXCEPT {
    auto const r_phase = static_cast<Rank>(phase);
    auto const r_nodes = static_cast<Rank>(nodes_);

    return isPow2(nodes_) ? rank_ xor r_phase : mod(rank_ + r_phase, r_nodes);
  }

  constexpr Rank recvRank(uint32_t phase) const FMPI_NOEXCEPT {
    auto const r_phase = static_cast<Rank>(phase);
    auto const r_nodes = static_cast<Rank>(nodes_);

    return isPow2(nodes_) ? rank_ xor r_phase : mod(rank_ - r_phase, r_nodes);
  }

  constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return nodes_;
  }

 private:
  Rank     rank_{};
  uint32_t nodes_{};
};

class OneFactor {
 public:
  static constexpr auto NAME = std::string_view("OneFactor");

  constexpr OneFactor() = default;

  constexpr OneFactor(Context const& ctx) FMPI_NOEXCEPT
    : OneFactor(ctx.size(), ctx.rank()) {
  }

  constexpr OneFactor(uint32_t nodes, Rank rank) FMPI_NOEXCEPT
    : rank_(rank),
      nodes_(nodes) {
  }

  constexpr Rank sendRank(uint32_t phase) const FMPI_NOEXCEPT {
    return (nodes_ % 2) != 0U ? factor_odd(phase) : factor_even(phase);
  }

  constexpr Rank recvRank(uint32_t phase) const FMPI_NOEXCEPT {
    return sendRank(phase);
  }

  constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return (nodes_ % 2) != 0U ? nodes_ : nodes_ - 1;
  }

 private:
  constexpr Rank factor_even(uint32_t phase) const FMPI_NOEXCEPT {
    auto const count = static_cast<Rank>(nodes_ - 1);
    auto const idle  = mod(static_cast<Rank>(nodes_ * phase / 2), count);

    if (rank_ == count) {
      return idle;
    }

    if (rank_ == idle) {
      return count;
    }

    return mod(static_cast<Rank>(phase) - rank_, count);
  }

  constexpr Rank factor_odd(uint32_t phase) const FMPI_NOEXCEPT {
    return mod(static_cast<Rank>(phase) - rank_, static_cast<Rank>(nodes_));
  }

  Rank     rank_{};
  uint32_t nodes_{};
};

class Linear {
 public:
  static constexpr auto NAME = std::string_view("Linear");

  static auto sendRank(Context const& comm, uint32_t phase) noexcept -> Rank;

  static auto recvRank(Context const& comm, uint32_t phase) noexcept -> Rank;

  static auto phaseCount(Context const& comm) noexcept -> uint32_t;
};

}  // namespace fmpi

#endif
