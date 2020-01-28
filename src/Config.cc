#include <omp.h>

#include <thread>

#include <fmpi/Config.hpp>
#include <fmpi/Debug.hpp>

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

  std::size_t domain_size = 1;
  {
    auto const* env = std::getenv("FMPI_DOMAIN_SIZE");
    if (env) {
      std::istringstream{std::string(env)} >> domain_size;
    }
  }

  auto const nthreads = std::thread::hardware_concurrency();

  if (nthreads < 4) {
    throw std::runtime_error("4 Threads at least required");
  }

  auto const ncores = nthreads / 2;

  auto const my_core         = sched_getcpu();
  auto const domain_id       = (my_core % ncores) / domain_size;
  auto const is_rank_on_comm = (my_core % domain_size) == 0;

  if (domain_size == 1) {
    dispatcher_core = (my_core + 1) % nthreads;
    scheduler_core  = (my_core + 2) % nthreads;
    comp_core       = my_core;
  } else if (is_rank_on_comm) {
    // MPI rank is on the communication core. So we dispatch on the other
    // hyperthread
    dispatcher_core =
        (std::size_t(my_core) < ncores) ? my_core + ncores : my_core - ncores;
    scheduler_core = my_core;
    comp_core      = scheduler_core + 1;
  } else {
    // MPI rank is somewhere in the Computation Domain, so we just take the
    // communication core.
    dispatcher_core = domain_id * domain_size;
    scheduler_core  = dispatcher_core + ncores;
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

  os << "Configuration:\n";

  os << "  " << std::setw(width) << std::left
     << "Main Core: " << config.main_core << "\n";
  os << "  " << std::setw(width) << std::left
     << "Dispatcher Core: " << config.dispatcher_core << "\n";
  os << "  " << std::setw(width) << std::left
     << "Scheduler Core: " << config.scheduler_core << "\n";
  os << "  " << std::setw(width) << std::left
     << "Computation Core: " << config.comp_core << "\n";

  auto const nthreads = omp_get_max_threads();
  auto const nplaces  = omp_get_num_places();

  std::ostringstream os_;
  os_ << "OMP Places [" << nthreads << ", " << nplaces << "]";

  os << "  " << std::setw(width) << std::left << os_.str();

  if (nplaces > 0) {
    os << ": {";

    std::vector<int> thread_ids(omp_get_max_threads());

    std::vector<int> myprocs;
    for (int i = 0; i < nplaces; ++i) {
      int nprocs = omp_get_place_num_procs(i);
      myprocs.resize(nprocs);
      omp_get_place_proc_ids(i, myprocs.data());
      std::copy(
          myprocs.begin(),
          std::prev(myprocs.end()),
          std::ostream_iterator<int>(os, ", "));
      os << *std::prev(myprocs.end());
    }

    os << "}\n";
  }

  os << std::endl;
}
