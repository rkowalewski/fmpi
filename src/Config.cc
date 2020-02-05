#include <omp.h>

#include <iomanip>
#include <iosfwd>
#include <thread>
#include <vector>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>

static inline int get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return std::thread::hardware_concurrency();
#endif
}

fmpi::Config const& fmpi::Config::instance() {
  static fmpi::Config config{};
  return config;
}
fmpi::Config::Config() {
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

  domain_size = 1;
  {
    auto const* env = std::getenv("FMPI_DOMAIN_SIZE");
    if (env) {
      std::istringstream{std::string(env)} >> domain_size;
    }
  }

  num_threads = get_num_threads();

  if (num_threads < 4) {
    throw std::runtime_error("4 Threads at least required");
  }

  auto const ncpus = num_threads / 2;

  auto const my_core         = sched_getcpu();
  auto const domain_id       = (my_core % ncpus) / domain_size;
  auto const is_rank_on_comm = (my_core % domain_size) == 0;

  if (domain_size == 1) {
    dispatcher_core = (my_core + 1) % num_threads;
    scheduler_core  = (my_core + 2) % num_threads;
    comp_core       = my_core;
  } else if (is_rank_on_comm) {
    // MPI rank is on the communication core. So we dispatch on the other
    // hyperthread
    dispatcher_core = (my_core < ncpus) ? my_core + ncpus : my_core - ncpus;
    scheduler_core  = my_core;
    comp_core       = my_core;
    // comp_core      = scheduler_core + 1;
  } else {
    // MPI rank is somewhere in the Computation Domain, so we just take the
    // communication core.
    dispatcher_core = domain_id * domain_size;
    scheduler_core  = dispatcher_core + ncpus;
    comp_core       = my_core;
  }

  main_core = my_core;
}

std::ostream& fmpi::operator<<(std::ostream& os, const Config& pinning) {
  os << "{ rank: " << pinning.main_core;
  os << ", scheduler: " << pinning.scheduler_core;
  os << ", dispatcher: " << pinning.dispatcher_core;
  os << ", comp: " << pinning.comp_core;
  os << " }";
  return os;
}

void fmpi::print_config(std::ostream& os) {
  auto const& config = Config::instance();

  constexpr int width = 20;

  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Main Core: " << config.main_core << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Dispatcher Core: " << config.dispatcher_core << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Scheduler Core: " << config.scheduler_core << "\n";
  os << "  " << std::setw(width) << std::left << std::setw(5) << " "
     << "Computation Core: " << config.comp_core << "\n";

  auto const nthreads = config.num_threads;
#ifdef _OPENMP
  auto const nplaces = omp_get_num_places();
#else
  auto const nplaces = 0;
#endif

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

    os << " ]\n";
  }
}
