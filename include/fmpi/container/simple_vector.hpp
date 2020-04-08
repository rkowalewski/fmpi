#ifndef CONTAINER__SIMPLE_VECTOR_H
#define CONTAINER__SIMPLE_VECTOR_H
#if 0

#include <tlx/container/simple_vector.hpp>

namespace fmpi {
template <class T>
class SimpleVector
  : public tlx::SimpleVector<T, SimpleVectorMode::NoInitNoDestroy> {
  using base_t = tlx::SimpleVector<T, SimpleVectorMode::NoInitNoDestroy>;

 public:
  using base_t::SimpleVector;

  void push_back(typename base_t::value_type const& v) noexcept
  {
    this->operator[m_nels++] = v;
  }

  void push_back(typename base_t::value_type&& v) noexcept
  {
    this->operator[m_nels++] = std::move(v);
  }

  void clear()
  {
    for (size_t i = 0; i < this->size(); ++i) {
      this->operator[i].~value_type();
    }

    m_nels = 0;
  }

 private:
  typename base_t::size_type m_nels{};
};

}  // namespace fmpi
#endif
#endif
