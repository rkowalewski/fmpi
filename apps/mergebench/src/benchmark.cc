#include "benchmark.hpp"

#include <fmpi/mpi/Environment.hpp>

#include <sstream>

std::ostream& operator<<(std::ostream& os, Params const& p) {
  std::ostringstream ss;

  ss << "{nprocs: " << p.nprocs << ", ";
  ss << "nblocks: " << p.nblocks << ", ";
  ss << "blocksz: " << p.blocksz << ", ";
  ss << "windowsz: " << p.windowsz << ", ";
  ss << "arraysize: " << p.arraysize << "}\n";

  os << ss.str();

  return os;
}

Params processParams(benchmark::State const& state) {
  Params params;

  params.nprocs   = state.range(0);
  params.blocksz  = state.range(1);
  params.windowsz = state.range(2);

  params.nblocks = params.nprocs /* * params.nprocs*/;

  params.arraysize = params.nblocks * params.blocksz;

  FMPI_DBG(params);

  return params;
}


// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
 public:
  NullReporter() = default;
  bool ReportContext(const Context&) override {
    return true;
  }
  void ReportRuns(const std::vector<Run>&) override {
  }
  void Finalize() override {
  }
};

// The main is rewritten to allow for MPI initializing and for selecting a
// reporter according to the process rank
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);

  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Single);

  auto const& world = mpi::Context::world();

  if (world.rank() == 0)
    // root process will use a reporter from the usual set provided by
    // ::benchmark
    ::benchmark::RunSpecifiedBenchmarks();
  else {
    // reporting from other processes is disabled by passing a custom
    // reporter
    NullReporter null;
    ::benchmark::RunSpecifiedBenchmarks(&null);
  }

  mpi::finalize();

  return 0;
}
