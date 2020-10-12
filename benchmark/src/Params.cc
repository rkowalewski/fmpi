#include <Version.h>
#include <unistd.h>

#include <Params.hpp>
#include <climits>
#include <cmath>
#include <ctime>
#include <fmpi/mpi/Environment.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <tlx/cmdline_parser.hpp>
#include <tlx/math/integer_log2.hpp>

namespace benchmark {

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

using onullstream = detail::basic_onullstream<char>;
// using wonullstream = detail::basic_onullstream<wchar_t>;

static auto getexepath() -> std::string {
  std::array<char, PATH_MAX> result{};
  ssize_t count = readlink("/proc/self/exe", result.data(), PATH_MAX);
  return std::string(result.data(), (count > 0) ? count : 0);
}

bool read_input(int argc, char* argv[], struct Params& params) {
  bool good = 0;

  tlx::CmdlineParser cp;

  // add description and author
  cp.set_description("Benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_string(
      'a',
      "algos",
      params.pattern,
      "Select specific algorithms matching a regex pattern");

  cp.add_bytes(
      's', "smin", params.smin, "Minimum message size sent to each peer.");
  cp.add_bytes(
      'S', "smax", params.smax, "Maximum message size sent to each peer.");

  cp.add_uint('p', "pmin", params.pmin, "minimum number of ranks");
  cp.add_uint('P', "pmax", params.pmax, "maximum number of ranks");

  cp.add_uint('i', "iterations", params.niters, "Trials per round.");

  cp.add_uint('w', "warmups", params.warmups, "Warmups per round.");

  std::size_t time_limit = params.time_limit.count();

  cp.add_size_t('t', "time_limit", time_limit, "maximum time limit (us)");

  params.time_limit = std::chrono::microseconds{time_limit};

  cp.add_flag(
      'c',
      "check",
      params.check,
      "Check if the SA has been constructed "
      "correctly. This does not work with random text (no way to "
      " reproduce).");

  auto const me = mpi::Context::world().rank();

  if (me == 0) {
    good = cp.process(argc, argv, std::cout);
  } else {
    onullstream os;
    good = cp.process(argc, argv, os);
  }

  if (good) {
    std::int64_t const nsteps = tlx::integer_log2_ceil(params.smax) -
                                tlx::integer_log2_ceil(params.smin);

    if (nsteps < 0) {
      if (me == 0) {
        std::ostringstream os;
        os << "ERROR: smax cannot be smaller than smin\n\n";
        cp.print_usage(os);
        std::cerr << os.str();
      }
      return false;
    }

    if (me == 0) {
      auto time = std::time(nullptr);
      std::cout << "Executable: " << getexepath() << "\n";
      std::cout << "Time: " << std::put_time(std::gmtime(&time), "%F %T")
                << "\n";
      std::cout << "Git Version: " << FMPI_GIT_COMMIT << "\n";

      cp.print_result();
      std::cout << "\n";
    }
  }

  return good;
}

static auto ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string& {
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

static auto rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string& {
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

static auto trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string& {
  return ltrim(rtrim(str, chars), chars);
}

struct string_pair : std::pair<std::string, std::string> {
  using std::pair<std::string, std::string>::pair;
};

auto operator<<(std::ostream& os, string_pair const& p) -> std::ostream& {
  os << p.first << " = " << p.second;
  return os;
}

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim) {
  std::vector<string_pair> envs;
  std::ostringstream       oss;
  for (auto** env = environ; *env != nullptr; ++env) {
    std::string var   = *env;
    auto        split = var.find('=');
    if (split != std::string::npos &&
        (var.find("OMPI_") != std::string::npos ||
         var.find("I_MPI") != std::string::npos ||
         var.find("OMP_") != std::string::npos ||
         var.find("FMPI_") != std::string::npos)) {
      auto key = var.substr(0, split);
      auto val = var.substr(split + 1);
      trim(key);
      trim(val);
      envs.emplace_back(key, val);
    }
  }

  for (auto&& kv : envs) {
    oss << prefix << kv << delim;
  }

  os << oss.str();
}

Params::Params() noexcept
  :
#ifdef NDEBUG
  niters(10)
  , warmups(1)
  , check(false)
#else
  niters(1)
  , warmups(0)
  , check(true)
#endif
{
}

void write_csv_header(std::ostream& os) {
  // os << "Nodes, Procs, Threads, Round, NBytes, Blocksize, Algo, Rank, "
  os << "Nodes, Procs, Threads, NBytes, Blocksize, Algo, Rank, "
        "Measurement, "
        "Value\n";
}

#if 0
std::ostream& operator<<(
    std::ostream& os, typename fmpi::TraceStore::mapped_type const& v) {
  std::visit([&os](auto const& val) { os << val; }, v);
  return os;
}
#endif

template <class Rep, class Period>
std::ostream& operator<<(
    std::ostream& os, const std::chrono::duration<Rep, Period>& d) {
  os << rtlx::to_seconds(d);
  return os;
}

void write_csv(
    std::ostream& os, Measurement const& params, Times const& times) {
  for (auto&& entry : times.traces) {
    std::ostringstream myos;
    myos << params.nhosts << ", ";
    myos << params.nprocs << ", ";
    myos << params.nthreads << ", ";
    // myos << params.step << ", ";
    myos << params.nbytes << ", ";
    myos << params.blocksize << ", ";
    myos << params.algorithm << ", ";
    myos << params.me << ", ";
    // myos << params.iter << ", ";
    myos << entry.first << ", ";
    myos << entry.second << "\n";
    os << myos.str();
  }
}

bool operator<(const Times& lhs, const Times& rhs) {
  return lhs.total_time < rhs.total_time;
}

}  // namespace benchmark
