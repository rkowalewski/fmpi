#include <Params.h>

#include <tlx/cmdline_parser.hpp>

namespace fmpi::benchmark {

bool process(
    int argc, char* argv[], ::mpi::Context const& mpiCtx, Params& params)
{
  tlx::CmdlineParser cp;

  // add description and author
  cp.set_description("Benchmark for the FMPI Algorithms Library.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_param_unsigned("nodes", params.nhosts, "Number of computation nodes");

#if 0
  std::string selected_algo = "";
  cp.add_opt_param_string(
      "algo", selected_algo, "Select a specific algorithm");
#endif

  std::array<size_t, 2> blocksizes = {params.minblocksize,
                                      params.maxblocksize};

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

  int good = false;

  // process command line
  if (mpiCtx.rank() == 0) {
    good = cp.process(argc, argv);
  }

  MPI_Bcast(&good, 1, mpi::type_mapper<int>::type(), 0, mpiCtx.mpiComm());

  if (!good) {
    return false;
  }

  if (mpiCtx.rank() == 0) {
    cp.print_result();
  }

  MPI_Bcast(
      &params.nhosts, 1, mpi::type_mapper<int>::type(), 0, mpiCtx.mpiComm());
  MPI_Bcast(
      &blocksizes[0],
      2,
      mpi::type_mapper<std::size_t>::type(),
      0,
      mpiCtx.mpiComm());

  params.minblocksize = blocksizes[0];
  params.maxblocksize = blocksizes[1];

  return true;
}
}  // namespace fmpi::benchmark
