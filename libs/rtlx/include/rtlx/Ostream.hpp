#ifndef RTLX_OSTREAM_HPP
#define RTLX_OSTREAM_HPP

#include <iosfwd>

namespace rtlx {
namespace detail {
template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf : public std::basic_streambuf<cT, traits> {
  auto overflow(typename traits::int_type c) ->
      typename traits::int_type override {
    return traits::not_eof(c);  // indicate success
  }
};

template <class cT, class traits = std::char_traits<cT> >
class basic_onullstream : public std::basic_ostream<cT, traits> {
 public:
  basic_onullstream()
    : std::basic_ios<cT, traits>(&m_sbuf)
    , std::basic_ostream<cT, traits>(&m_sbuf) {
    // note: the original code is missing the required this->
    this->init(&m_sbuf);
  }

 private:
  basic_nullbuf<cT, traits> m_sbuf;
};
}  // namespace detail

using onullstream  = detail::basic_onullstream<char>;
using wonullstream = detail::basic_onullstream<wchar_t>;
}  // namespace rtlx
#endif
