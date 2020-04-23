#ifndef FMPI_CONTAINER_STACKCONTAINER_HPP
#define FMPI_CONTAINER_STACKCONTAINER_HPP

#include <vector>

#include <fmpi/Constants.hpp>

#include <tlx/stack_allocator.hpp>

namespace fmpi {

namespace detail {

template <class T, std::size_t N>
constexpr std::size_t stackCapacity() {
  constexpr auto requested = N * sizeof(T);

  return (kContainerStackSize < requested) ? kContainerStackSize : requested;
}
}  // namespace detail

template <class ContainerType, std::size_t Capacity>
class StackContainer {
  static_assert(Capacity <= kContainerStackSize, "invalid stack size");
  using stack_arena = tlx::StackArena<Capacity>;

 public:
  using container_type = ContainerType;
  using value_type     = typename container_type::value_type;
  using allocator_type = tlx::StackAllocator<value_type, Capacity>;

  StackContainer()
    : allocator_(arena_)
    , container_(allocator_) {
  }

  // Getters for the actual container.
  //
  // Danger: any copies of this made using the copy constructor must have
  // shorter lifetimes than the source. The copy will share the same allocator
  // and therefore the same stack buffer as the original. Use std::copy to
  // copy into a "real" container for longer-lived objects.
  container_type& container() {
    return container_;
  }
  [[nodiscard]] const container_type& container() const {
    return container_;
  }

  ContainerType* operator->() {
    return &container_;
  }
  const ContainerType* operator->() const {
    return &container_;
  }

  StackContainer(const StackContainer&) = delete;
  StackContainer& operator=(const StackContainer&) = delete;

 protected:
  stack_arena    arena_{};
  allocator_type allocator_;
  container_type container_;
};

template <class T, std::size_t Size>
class StackVector
  : public StackContainer<
        std::vector<
            T,
            tlx::StackAllocator<T, detail::stackCapacity<T, Size>()>>,
        detail::stackCapacity<T, Size>()> {
 private:
  static constexpr std::size_t capacity = detail::stackCapacity<T, Size>();

  using base = StackContainer<
      std::vector<T, tlx::StackAllocator<T, capacity>>,
      capacity>;

 public:
  StackVector()
    : base() {
  }

  // Copy Constructor if we want to put this into a different STL container
  StackVector(const StackVector<T, Size>& other)
    : base() {
    this->container().assign(other->begin(), other->end());
  }

  StackVector<T, Size>& operator=(const StackVector<T, Size>& other) {
    this->container().assign(other->begin(), other->end());
    return *this;
  }

  // just for convenience
  T& operator[](size_t i) {
    return this->container().operator[](i);
  }
  const T& operator[](size_t i) const {
    return this->container().operator[](i);
  }
};

}  // namespace fmpi

#endif
