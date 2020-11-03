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
  explicit Schedule(const T& obj)
    : object(std::make_shared<Model<T>>(std::move(obj))) {
  }

  [[nodiscard]] std::string_view name() const {
    return object->name();
  }

  [[nodiscard]] mpi::Rank sendRank(uint32_t p) const {
    return object->sendRank(p);
  }

  [[nodiscard]] mpi::Rank recvRank(uint32_t p) const {
    return object->recvRank(p);
  }

  [[nodiscard]] uint32_t phaseCount() const {
    return object->phaseCount();
  }

  [[nodiscard]] bool is_intermediate() const {
    return object->is_intermediate();
  }

 private:
  struct Concept {
    virtual ~Concept() {
    }
    [[nodiscard]] virtual std::string_view name() const               = 0;
    [[nodiscard]] virtual mpi::Rank        sendRank(uint32_t p) const = 0;
    [[nodiscard]] virtual mpi::Rank        recvRank(uint32_t p) const = 0;
    [[nodiscard]] virtual uint32_t         phaseCount() const         = 0;
    [[nodiscard]] virtual bool             is_intermediate() const    = 0;
  };

  template <typename T>
  struct Model : Concept {
    explicit Model(T t)
      : object(std::move(t)) {
    }
    [[nodiscard]] std::string_view name() const override {
      return object.name();
    }

    [[nodiscard]] mpi::Rank sendRank(uint32_t p) const override {
      return object.sendRank(p);
    }

    [[nodiscard]] mpi::Rank recvRank(uint32_t p) const override {
      return object.recvRank(p);
    }

    [[nodiscard]] uint32_t phaseCount() const override {
      return object.phaseCount();
    }

    [[nodiscard]] bool is_intermediate() const override {
      return object.is_intermediate();
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

  static constexpr std::string_view name() noexcept {
    return std::string_view("Ring");
  }

  static constexpr bool is_intermediate() noexcept {
    return false;
  }

 private:
  Rank const     rank_{};
  uint32_t const nodes_{};
};

class OneFactor {
  using Rank    = mpi::Rank;
  using Context = mpi::Context;

 public:
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

  static constexpr std::string_view name() noexcept {
    return std::string_view("OneFactor");
  }

  static constexpr bool is_intermediate() noexcept {
    return false;
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
  constexpr Linear() = default;

  constexpr explicit Linear(Context const& ctx) FMPI_NOEXCEPT
    : Linear(ctx.size(), ctx.rank()) {
  }

  constexpr Linear(uint32_t nodes, Rank rank) FMPI_NOEXCEPT : rank_(rank),
                                                              nodes_(nodes) {
  }

  [[nodiscard]] static constexpr Rank sendRank(uint32_t phase) FMPI_NOEXCEPT {
    return Rank{static_cast<int>(phase)};
  }

  [[nodiscard]] static constexpr Rank recvRank(uint32_t phase) FMPI_NOEXCEPT {
    return Rank{static_cast<int>(phase)};
  }

  [[nodiscard]] constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return nodes_;
  }

  static constexpr std::string_view name() noexcept {
    return std::string_view("Linear");
  }

  static constexpr bool is_intermediate() noexcept {
    return false;
  }

 private:
  Rank const     rank_{};
  uint32_t const nodes_{};
};

class Bruck {
  using Rank    = mpi::Rank;
  using Context = mpi::Context;

 public:
  constexpr Bruck() = default;

  constexpr explicit Bruck(Context const& ctx) FMPI_NOEXCEPT
    : Bruck(ctx.size(), ctx.rank()) {
  }

  constexpr Bruck(uint32_t nodes, Rank rank) FMPI_NOEXCEPT : rank_(rank),
                                                             nodes_(nodes) {
  }

  [[nodiscard]] static constexpr Rank sendRank(uint32_t phase) FMPI_NOEXCEPT {
    return Rank{static_cast<int>(phase)};
  }

  [[nodiscard]] static constexpr Rank recvRank(uint32_t phase) FMPI_NOEXCEPT {
    return Rank{static_cast<int>(phase)};
  }

  [[nodiscard]] constexpr uint32_t phaseCount() const FMPI_NOEXCEPT {
    return nodes_;
  }

  static constexpr std::string_view name() noexcept {
    return std::string_view("Bruck");
  }

  static constexpr bool is_intermediate() noexcept {
    return true;
  }

 private:
  Rank const     rank_{};
  uint32_t const nodes_{};
};

struct ScheduleOpts {
  enum class WindowType
  {
    sliding,
    fixed
  };

  template <class Schedule>
  ScheduleOpts(
      Schedule         schedule_,
      uint32_t         winsz_,
      std::string_view name_,
      WindowType       type_)
    : schedule(schedule_)
    , winsz(std::min(schedule.phaseCount(), winsz_))
    , type(type_)
#ifndef NDEBUG
    , name(name_)
#endif
  {
    std::ignore = name_;
  }

  detail::Schedule const schedule;
  uint32_t const         winsz = 0;
  WindowType const       type  = WindowType::fixed;
#ifndef NDEBUG
  std::string const name;
#endif
};

}  // namespace fmpi

#endif
