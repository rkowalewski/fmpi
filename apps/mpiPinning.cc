#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char** argv)
{
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  // rank
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // processor name
  char _hostname[MPI_MAX_PROCESSOR_NAME];
  int  resultlen;
  MPI_Get_processor_name(_hostname, &resultlen);
  std::string hostname(_hostname);

  std::vector<int> cpu_names(omp_get_max_threads());
  int              mpi_cpu = sched_getcpu();

#pragma omp parallel default(none) shared(cpu_names)
  {
    auto thread_num       = omp_get_thread_num();  // Test
    int  cpu              = sched_getcpu();
    cpu_names[thread_num] = cpu;
  }

  if (me == 0) {
    std::cout << "Rank, CPU, RankCore, Threads, ThreadID, Core\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::ostringstream os;
  for (std::size_t i = 0; i < cpu_names.size(); ++i) {
    os << me << ", " << hostname << ", " << mpi_cpu << ", "
       << cpu_names.size() << ", " << i << ", " << cpu_names[i] << "\n";
  }

  std::cout << os.str();

  MPI_Finalize();

  return 0;
}
