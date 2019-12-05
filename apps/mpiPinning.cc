#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <iostream>
#include <sstream>

#include <fmpi/mpi/Environment.hpp>
#include <fmpi/NumericRange.hpp>

#include <rtlx/Assert.hpp>
#include <rtlx/ScopedLambda.hpp>

int main(int argc, char** argv)
{
  //MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);


  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::Context worldCtx{MPI_COMM_WORLD};

  auto const me = worldCtx.rank();
  auto const nr = worldCtx.size();

  char _hostname[MPI_MAX_PROCESSOR_NAME];
  int  resultlen;

  RTLX_ASSERT_RETURNS(
      MPI_Get_processor_name(_hostname, &resultlen), MPI_SUCCESS);

  std::string hostname(_hostname);

  std::vector<int> cpu_names(omp_get_max_threads());

  {
    std::ostringstream os;
    os << "last cpu: " << worldCtx.getLastCPU() << "\n";
    std::cout << os.str();
  }


#pragma omp parallel default(none) shared(cpu_names)
  {
    auto thread_num = omp_get_thread_num(); //Test
    int cpu = sched_getcpu();
    cpu_names[thread_num] = cpu;
  }


  if (me == 0) {
    std::cout << "Rank, CPU, Threads, ThreadID, Core\n";
  }

  MPI_Barrier(worldCtx.mpiComm());

  std::ostringstream os;
  for (auto&& r : fmpi::range(cpu_names.size())) {
    os << me << ", "
      << hostname << ", "
      << cpu_names.size() << ", "
      << r << ", "
      << cpu_names[r] << "\n";
  }

  std::cout << os.str();


  return 0;
}
