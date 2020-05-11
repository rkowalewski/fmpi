#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <fmpi/concurrency/CacheLocality.hpp>

int main(int argc, char** argv) {
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

  int nplaces = omp_get_num_places();
  printf("omp_get_num_places: %d\n", nplaces);
  printf("omp_get_max_threads: %d\n", omp_get_max_threads());

  std::vector<int> myprocs;
  for (int i = 0; i < nplaces; ++i) {
    int nprocs = omp_get_place_num_procs(i);
    myprocs.resize(nprocs);
    omp_get_place_proc_ids(i, myprocs.data());
    std::ostringstream os;
    os << "rank: " << me << ", place num: " << i << ", places: {";
    std::copy(
        myprocs.begin(), myprocs.end(), std::ostream_iterator<int>(os, ", "));
    os << "}\n";
    std::cout << os.str();
  }
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
    os << me << ", " << hostname << ", " << sched_getcpu() << ", "
       << cpu_names.size() << ", " << i << ", " << cpu_names[i] << "\n";
  }

  std::cout << os.str();

  MPI_Finalize();

  {
    auto& sys = folly::CacheLocality::system<>();

    std::ostringstream os1;

    os1 << "numCpus= " << sys.numCpus << "\n";
    os1 << "numCachesByLevel= \n";
    for (std::size_t i = 0; i < sys.numCachesByLevel.size(); ++i) {
      os1 << "  [" << i << "]= " << sys.numCachesByLevel[i] << "\n";
    }
    os1 << "localityIndexByCpu= \n";
    for (std::size_t i = 0; i < sys.localityIndexByCpu.size(); ++i) {
      os1 << "  [" << i << "]= " << sys.localityIndexByCpu[i] << "\n";
    }

    if (me == 0) {
      std::cout << os1.str();
    }
  }

  return 0;
}
