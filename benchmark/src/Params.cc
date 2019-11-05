#include <Params.h>

#include <tlx/cmdline_parser.hpp>

#include <rtlx/Assert.h>

#include <Version.h>

extern char** environ;

namespace fmpi {
namespace benchmark {

bool process(
    int argc, char* argv[], ::mpi::Context const& mpiCtx, Params& params)
{
  bool good;

  std::array<size_t, 2> blocksizes = {params.minblocksize,
                                      params.maxblocksize};

  if (mpiCtx.rank() == 0) {
    tlx::CmdlineParser cp;

    // add description and author
    cp.set_description("Benchmark for the FMPI Algorithms Library.");
    cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

    cp.add_param_unsigned(
        "nodes", params.nhosts, "Number of computation nodes");

#if 0
    std::string selected_algo = "";
    cp.add_opt_param_string(
        "algo", selected_algo, "Select a specific algorithm");
#endif

    cp.add_bytes(
        'l',
        "minblocksize",
        blocksizes[0],
        "Minimum block size communication to each unit.");
    cp.add_bytes(
        'u',
        "maxblocksize",
        blocksizes[1],
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

    good = cp.process(argc, argv);

    cp.print_result();
  }

  RTLX_ASSERT_RETURNS(
      MPI_Bcast(
          &good, 1, mpi::type_mapper<bool>::type(), 0, mpiCtx.mpiComm()),
      MPI_SUCCESS);

  if (static_cast<int>(good) == 0) {
    return false;
  }

  RTLX_ASSERT_RETURNS(
      MPI_Bcast(
          &params.nhosts,
          1,
          mpi::type_mapper<int>::type(),
          0,
          mpiCtx.mpiComm()),
      MPI_SUCCESS);

  RTLX_ASSERT_RETURNS(
      MPI_Bcast(
          &params.niters,
          1,
          mpi::type_mapper<int>::type(),
          0,
          mpiCtx.mpiComm()),
      MPI_SUCCESS);

  RTLX_ASSERT_RETURNS(
      MPI_Bcast(
          &params.check,
          1,
          mpi::type_mapper<bool>::type(),
          0,
          mpiCtx.mpiComm()),
      MPI_SUCCESS);

  RTLX_ASSERT_RETURNS(
      MPI_Bcast(
          &blocksizes[0],
          2,
          mpi::type_mapper<std::size_t>::type(),
          0,
          mpiCtx.mpiComm()),
      MPI_SUCCESS);

  params.minblocksize = blocksizes[0];
  params.maxblocksize = blocksizes[1];

  return true;
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
    std::ostream& os, std::string prefix, const char* delim)
{
  std::vector<string_pair> envs;
  std::ostringstream oss;
  for (auto** env = environ; *env != 0; ++env) {
    std::string var   = *env;
    auto        split = var.find('=');
    if (split != std::string::npos &&
        (var.find("OMPI_") != std::string::npos ||
         var.find("I_MPI") != std::string::npos)) {
      auto key = var.substr(0, split);
      auto val = var.substr(split + 1);
      trim(key);
      trim(val);
      envs.push_back(string_pair(key, val));
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
