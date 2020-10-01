#ifndef FMPI_CONTAINER_FIXED_VECTOR_HPP
#define FMPI_CONTAINER_FIXED_VECTOR_HPP

#include <vector>

namespace fmpi {

template <class T, class A = std::allocator<T>>
struct FixedVector : private std::vector<T, A> {
  using FixedVector::vector::vector;
  using FixedVector::vector::operator=;
  using FixedVector::vector::at;
  using FixedVector::vector::back;
  using FixedVector::vector::begin;
  using FixedVector::vector::cbegin;
  using FixedVector::vector::cend;
  using FixedVector::vector::data;
  using FixedVector::vector::empty;
  using FixedVector::vector::end;
  using FixedVector::vector::front;
  using FixedVector::vector::get_allocator;
  using FixedVector::vector::size;
  using FixedVector::vector::operator[];
};
}  // namespace fmpi

#endif
