#include <omp.h>

#include <iomanip>
#include <iosfwd>
#include <thread>
#include <vector>

#include <fmpi/Debug.hpp>
#include <fmpi/Pinning.hpp>
#include <fmpi/common/Porting.hpp>
#include <fmpi/mpi/Environment.hpp>

#include <fmpi/concurrency/CacheLocality.hpp>

static uint32_t num_world_nodes() {
  auto const& world = mpi::Context::world();
  MPI_Comm    shmcomm;
  int         shmrank;
  uint32_t    nodes;

  FMPI_CHECK_MPI(MPI_Comm_split_type(
      world.mpiComm(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm));

  FMPI_CHECK_MPI(MPI_Comm_rank(shmcomm, &shmrank));
  int const is_rank0 = (shmrank == 0) ? 1 : 0;
  FMPI_CHECK_MPI(
      MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, world.mpiComm()));
  FMPI_CHECK_MPI(MPI_Comm_free(&shmcomm));
  return nodes;
}

fmpi::Pinning const& fmpi::Pinning::instance() {
  static fmpi::Pinning config{};
  return config;
}

fmpi::Pinning::Pinning() {
  {
    int flag;
    FMPI_CHECK_MPI(MPI_Initialized(&flag));

    if (flag == 0) {
      throw std::runtime_error("MPI not initialized");
    }

    FMPI_CHECK_MPI(MPI_Is_thread_main(&flag));

    if (flag == 0) {
      throw std::runtime_error("Configuration only allowed from main thread");
    }
  }

  num_nodes = num_world_nodes();

  domain_size = 1;
  {
    auto const* env = std::getenv("FMPI_DOMAIN_SIZE");
    if (env != nullptr) {
      std::istringstream{std::string(env)} >> domain_size;
    }
  }

  num_threads = get_num_user_threads();

  auto const ncpus = folly::CacheLocality::system().numCpus / 2;

  main_core                  = sched_getcpu();
  auto const domain_id       = (main_core % ncpus) / domain_size;
  auto const is_rank_on_comm = (main_core % domain_size) == 0;

  if (domain_size == 1) {
    dispatcher_core = (main_core + 1) % num_threads;
    // scheduler_core  = (main_core + 2) % num_threads;
    // comp_core       = main_core;
  } else if (is_rank_on_comm) {
    // MPI rank is on the communication core. So we dispatch on the other
    // hyperthread
    dispatcher_core =
        (main_core < ncpus) ? main_core + ncpus : main_core - ncpus;
    // scheduler_core = main_core;
    // comp_core      = main_core;
  } else {
    // MPI rank is somewhere in the Computation Domain, so we just take the
    // communication core.
    dispatcher_core = domain_id * domain_size;
    // scheduler_core  = dispatcher_core + ncpus;
    // comp_core       = main_core;
  }
}

std::ostream& fmpi::operator<<(std::ostream& os, const Pinning& pinning) {
  os << "{ rank: " << pinning.main_core;
  os << ", dispatcher: " << pinning.dispatcher_core;
  // os << ", scheduler: " << pinning.scheduler_core;
  // os << ", comp: " << pinning.comp_core;
  os << " }";
  return os;
}

void fmpi::print_pinning(std::ostream& os) {
  auto const& config = Pinning::instance();

  constexpr int width = 20;

  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Main Core: " << config.main_core << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Dispatcher Core: " << config.dispatcher_core << "\n";
#if 0
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Scheduler Core: " << config.scheduler_core << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Computation Core: " << config.comp_core << "\n";
#endif

  auto const nthreads = config.num_threads;
#ifdef _OPENMP
  auto const nplaces = omp_get_num_places();
#else
  auto const nplaces = 0;
#endif

  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Domain size: " << config.domain_size << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Threads: " << nthreads << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Places: " << nplaces;

  if (nplaces > 0) {
    os << " [ ";
    std::vector<int> myprocs;
    for (int i = 0; i < nplaces; ++i) {
      int const nprocs = omp_get_place_num_procs(i);
      myprocs.resize(nprocs);
      omp_get_place_proc_ids(i, myprocs.data());

      std::ostringstream os1;
      os1 << "{";
      std::copy(
          myprocs.begin(),
          std::prev(myprocs.end()),
          std::ostream_iterator<int>(os1, ", "));
      os1 << *std::prev(myprocs.end());
      os1 << "}";
      if (i < (nplaces - 1)) {
        os1 << ", ";
      }
      os << os1.str();
    }

    os << "]";
  }

  os << std::endl;
}