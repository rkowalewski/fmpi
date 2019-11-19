#include <Params.h>

#include <tlx/cmdline_parser.hpp>

#include <rtlx/Assert.h>

#include <Version.h>

extern char** environ;

namespace fmpi {
namespace benchmark {

template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf : public std::basic_streambuf<cT, traits> {
  typename traits::int_type overflow(typename traits::int_type c) override
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

bool process(
    int                   argc,
    char*                 argv[],
    ::mpi::Context const& mpiCtx,
    struct Params&        params)
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

  cp.add_bytes(
      'l',
      "minblocksize",
      params.minblocksize,
      "Minimum block size communication to each unit.");
  cp.add_bytes(
      'u',
      "maxblocksize",
      params.maxblocksize,
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

  if (mpiCtx.rank() == 0) {
    good = cp.process(argc, argv, std::cout);
  }
  else {
    onullstream os;
    good = cp.process(argc, argv, os);
  }

  if (good && mpiCtx.rank() == 0) {
    cp.print_result();
  }

  return good;
}

static std::string& ltrim(
    std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

static std::string& rtrim(
    std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

static std::string& trim(
    std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
  return ltrim(rtrim(str, chars), chars);
}

struct string_pair : std::pair<std::string, std::string> {
  using std::pair<std::string, std::string>::pair;
};

std::ostream& operator<<(std::ostream& os, string_pair const& p)
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
