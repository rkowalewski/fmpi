#include <Params.h>
#include <Version.h>
#include <fmpi/NumericRange.h>
#include <rtlx/Assert.h>

#include <cmath>
#include <string>
#include <tlx/cmdline_parser.hpp>

extern char** environ;

namespace fmpi {
namespace benchmark {

template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf : public std::basic_streambuf<cT, traits> {
  auto overflow(typename traits::int_type c) ->
      typename traits::int_type override
  {
    return traits::not_eof(c);  // indicate success
  }
};

template <class cT, class traits = std::char_traits<cT> >
class basic_onullstream : public std::basic_ostream<cT, traits> {
 public:
  basic_onullstream()
    : std::basic_ios<cT, traits>(&m_sbuf)
    , std::basic_ostream<cT, traits>(&m_sbuf)
  {
    // note: the original code is missing the required this->
    this->init(&m_sbuf);
  }

 private:
  basic_nullbuf<cT, traits> m_sbuf;
};

using onullstream  = basic_onullstream<char>;
using wonullstream = basic_onullstream<wchar_t>;

auto process(
    int                   argc,
    char*                 argv[],
    ::mpi::Context const& mpiCtx,
    struct Params&        params) -> bool
{
  bool good;

  tlx::CmdlineParser cp;

  // add description and author
  cp.set_description("Benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_param_unsigned(
      "nodes", params.nhosts, "Number of computation nodes");

  cp.add_string(
      'a',
      "algos",
      params.pattern,
      "Select specific algorithms matching a regex pattern");

  std::size_t minblocksize;

  std::size_t maxblocksize;
  cp.add_bytes(
      'l',
      "minblocksize",
      minblocksize,
      "Minimum block size communication to each unit.");
  cp.add_bytes(
      'u',
      "maxblocksize",
      maxblocksize,
      "Maximum block size communication to each unit.");

  cp.add_uint(
      'i', "iterations", params.niters, "Number of iterations per round.");

  cp.add_flag(
      'c',
      "check",
      params.check,
      "Check if the SA has been constructed "
      "correctly. This does not work with random text (no way to "
      " reproduce).");

  std::string sizes_csv;
  cp.add_string('s', "sizes", sizes_csv, "list of block sizes");

  if (mpiCtx.rank() == 0) {
    good = cp.process(argc, argv, std::cout);
  }
  else {
    onullstream os;
    good = cp.process(argc, argv, os);
  }

  if (sizes_csv.empty()) {
    std::int64_t nsteps = std::ceil(std::log2(maxblocksize)) -
                          std::ceil(std::log2(minblocksize));

    if (nsteps < 0) {
      if (mpiCtx.rank() == 0) {
        std::ostringstream os;
        os << "maxblocksize cannot be smaller than minblocksize\n";
        cp.print_usage(os);
        std::cerr << os.str();
        return false;
      }
    }

    nsteps = std::min<std::size_t>(nsteps, 20);

    params.sizes.resize(nsteps + 1);

    auto blocksize = minblocksize;
    for (auto&& r : range<std::size_t>(nsteps + 1)) {
      params.sizes[r] = std::min(blocksize, maxblocksize);
      blocksize *= 2;
    }
  }
  else {
    std::stringstream ss(sizes_csv);

    for (std::size_t i; ss >> i;) {
      params.sizes.push_back(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }
  }

  if (good && mpiCtx.rank() == 0) {
    cp.print_result();
  }

  return good;
}

static auto ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string&
{
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

static auto rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string&
{
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

static auto trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
    -> std::string&
{
  return ltrim(rtrim(str, chars), chars);
}

struct string_pair : std::pair<std::string, std::string> {
  using std::pair<std::string, std::string>::pair;
};

auto operator<<(std::ostream& os, string_pair const& p) -> std::ostream&
{
  os << p.first << " = " << p.second;
  return os;
}

void printBenchmarkPreamble(
    std::ostream& os, const std::string& prefix, const char* delim)
{
  std::vector<string_pair> envs;
  std::ostringstream       oss;
  for (auto** env = environ; *env != nullptr; ++env) {
    std::string var   = *env;
    auto        split = var.find('=');
    if (split != std::string::npos &&
        (var.find("OMPI_") != std::string::npos ||
         var.find("I_MPI") != std::string::npos ||
         var.find("OMP_") != std::string::npos)) {
      auto key = var.substr(0, split);
      auto val = var.substr(split + 1);
      trim(key);
      trim(val);
      envs.emplace_back(key, val);
    }
  }

  oss << prefix << "Git Version: " << FMPI_GIT_COMMIT << delim;

  for (auto&& kv : envs) {
    oss << prefix << kv << delim;
  }

  os << oss.str();
}
}  // namespace benchmark
}  // namespace fmpi
