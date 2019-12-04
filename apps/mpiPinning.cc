#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <iostream>
#include <sstream>

#include <fmpi/mpi/Environment.hpp>

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

  std::ostringstream os;
  os << "I am " << hostname << " (rank " << me << " of " << nr << ")\n";

  std::cout << os.str();

#pragma omp parallel default(none) firstprivate(me)
  {
    auto thread_num = omp_get_thread_num(); //Test
    int cpu = sched_getcpu();

    printf("Rank: %d, ThreadId: %d, cpu: %d\n", static_cast<int>(me), thread_num, cpu);
  }

  return 0;
}
