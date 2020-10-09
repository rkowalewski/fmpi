#ifndef FMPI_SCHEDULE_HPP
#define FMPI_SCHEDULE_HPP

#include <fmpi/Config.hpp>
#include <fmpi/mpi/Environment.hpp>
#include <fmpi/util/Math.hpp>

namespace fmpi {

namespace detail {

class Schedule {
 public:
  template <typename T>
  Schedule(const T& obj)
    : object(std::make_shared<Model<T>>(std::move(obj))) {
  }

  std::string_view name() const {
    return object->name();
  }

  mpi::Rank sendRank(uint32_t p) const {
    return object->sendRank(p);
  }

  mpi::Rank recvRank(uint32_t p) const {
    return object->recvRank(p);
  }

  uint32_t phaseCount() const {
    return object->phaseCount();
  }

 private:
  struct Concept {
    virtual ~Concept() {
    }
    virtual std::string_view name() const               = 0;
    virtual mpi::Rank        sendRank(uint32_t p) const = 0;
    virtual mpi::Rank        recvRank(uint32_t p) const = 0;
    virtual uint32_t         phaseCount() const         = 0;
  };

  template <typename T>
  struct Model : Concept {
    Model(const T& t)
      : object(t) {
    }
    std::string_view name() const override {
      return T::NAME;
    }

    mpi::Rank sendRank(uint32_t p) const override {
      return object.sendRank(p);
    }

    mpi::Rank recvRank(uint32_t p) const override {
      return object.recvRank(p);
    }

    uint32_t phaseCount() const override {
      return object.phaseCount();
    }

   private:
    T object;
  };

  std::shared_ptr<const Concept> object;
};

}  // namespace detail

class FlatHandshake {
  using Rank    = mpi::Rank;
  using Context = mpi::Context;

 public:
  static constexpr auto NAME = std::string_view("Ring");

  constexpr FlatHandshake() = default;

  constexpr explicit FlatHandshake(Context const& ctx) FMPI_NOEXCEPT
    : FlatHandshake(ctx.size(), ctx.rank()) {
  }

  constexpr FlatHandshake(uint32_t nodes, Rank rank) FMPI_NOEXCEPT
    : rank_(rank),
      nodes_(nodes) {
  }

  [[nodiscard]] constexpr Rank sendRank(uint32_t phase) const FMPI_NOEXCEPT {
    auto const r_phase = static_cast<Rank>(phase);
    auto const r_nodes = static_cast<Rank>(nodes_);

    return isPow2(nodes_) ? rank_ xor r_phase : mod(rank_ + r_phase, r_nodes);
  }

  [[nodiscard]] constexpr Rank recvRank(uint32_t phase) const FMPI_NOEXCEPT {
    auto const r_phase = static_cast<Rank>(phase);
    auto const r_nodes = static_cast<Rank>(nodes_);

    return isPow2(nodes_) ? rank_ xor r_phase : mod(rank_ - r_phase, r_nodes);
  }

  [[nodiscard]] constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return nodes_;
  }

 private:
  Rank const     rank_{};
  uint32_t const nodes_{};
};

class OneFactor {
  using Rank    = mpi::Rank;
  using Context = mpi::Context;

 public:
  static constexpr auto NAME = std::string_view("OneFactor");

  constexpr OneFactor() = default;

  constexpr explicit OneFactor(Context const& ctx) FMPI_NOEXCEPT
    : OneFactor(ctx.size(), ctx.rank()) {
  }

  constexpr OneFactor(uint32_t nodes, Rank rank) FMPI_NOEXCEPT
    : rank_(rank),
      nodes_(nodes) {
  }

  [[nodiscard]] constexpr Rank sendRank(uint32_t phase) const FMPI_NOEXCEPT {
    return (nodes_ % 2) != 0U ? factor_odd(phase) : factor_even(phase);
  }

  [[nodiscard]] constexpr Rank recvRank(uint32_t phase) const FMPI_NOEXCEPT {
    return sendRank(phase);
  }

  [[nodiscard]] constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return (nodes_ % 2) != 0U ? nodes_ : nodes_ - 1;
  }

 private:
  [[nodiscard]] constexpr Rank factor_even(uint32_t phase) const
      FMPI_NOEXCEPT {
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

  [[nodiscard]] constexpr Rank factor_odd(uint32_t phase) const
      FMPI_NOEXCEPT {
    return mod(static_cast<Rank>(phase) - rank_, static_cast<Rank>(nodes_));
  }

  Rank const     rank_{};
  uint32_t const nodes_{};
};

class Linear {
  using Rank    = mpi::Rank;
  using Context = mpi::Context;

 public:
  static constexpr auto NAME = std::string_view("Linear");

  static auto sendRank(Context const& comm, uint32_t phase) noexcept -> Rank;

  static auto recvRank(Context const& comm, uint32_t phase) noexcept -> Rank;

  static auto phaseCount(Context const& comm) noexcept -> uint32_t;
};

struct ScheduleOpts {
  enum class WindowType
  {
    sliding,
    fixed
  };

  template <class Schedule>
  ScheduleOpts(
      Schedule         schedule,
      std::size_t      winsz_,
      std::string_view name_,
      WindowType       type_)
    : scheduler(schedule)
    , winsz(winsz_)
    , name(name_)
    , type(type_) {
  }

  detail::Schedule const scheduler;
  std::size_t const      winsz = 0;
  std::string_view       name;
  WindowType const       type = WindowType::fixed;
};

}  // namespace fmpi

#endif
