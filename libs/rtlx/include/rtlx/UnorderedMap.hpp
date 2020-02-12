#ifndef RTLX_ALGORITHMS_HPP
#define RTLX_ALGORITHMS_HPP

#include <unordered_map>

namespace rtlx {

template <
    class Key,
    class T,
    class Hash,
    class KeyEqual,
    class Alloc,
    class Pred>
void erase_if(
    std::unordered_map<Key, T, Hash, KeyEqual, Alloc>& c, Pred pred) {
  for (auto it = std::begin(c); it != std::end(c);) {
    if (pred(*it)) {
      it = c.erase(it);
    } else {
      ++it;
    }
  }
}
}  // namespace rtlx

#endif
