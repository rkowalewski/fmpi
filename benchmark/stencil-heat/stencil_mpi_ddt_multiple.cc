/*
 * Copyright (c) 2012 Torsten Hoefler. All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@illinois.edu>
 *
 */

#include <omp.h>

#include <fmpi/Debug.hpp>
#include <iostream>
#include <rtlx/Ostream.hpp>
#include <tlx/cmdline_parser.hpp>

#include "stencil_par.h"

#define THX_START \
  (thread_id % nthreads) == 0 ? 1 : (thread_id % nthreads) * Thx + 1
#define THX_END                          \
  (thread_id % nthreads) == nthreads - 1 \
      ? bx + 1                           \
      : ((thread_id + 1) % nthreads) * Thx + 1

void init_sources(
    int       bx,
    int       by,
    int       offx,
    int       offy,
    int       n,
    const int nsources,
    int       sources[][2],
    int*      locnsources_ptr,
    int       locsources[][2]);

struct ThreadComm {
  int west, east, north, south;
};

void neighbor_alltoallv(
    const void*          sbuf,
    const int*           scounts,
    const int*           sdispls,
    MPI_Datatype         stype,
    void*                rbuf,
    const int*           rcounts,
    const int*           rdispls,
    MPI_Datatype         rtype,
    mpi::Context const&  comm,
    fmpi::ScheduleHandle hdl) {
}

int main(int argc, char** argv) {
  int n, energy, niters, px, py;

  int rx, ry;
  int north, south, west, east;
  int bx, by, offx, offy;

  std::string fname;

  /* three heat sources */
  const int nsources = 3;
  int       sources[nsources][2];
  int       locnsources;             /* number of sources in my area */
  int       locsources[nsources][2]; /* sources local to my rank */

  double t1, t2;

  int iter, i, j;

  double heat, rheat;

  int nthreads, Thx;

  /* initialize MPI envrionment */
  mpi::initialize(&argc, &argv, mpi::ThreadLevel::Serialized);
  auto finalizer = rtlx::scope_exit([]() { mpi::finalize(); });

  auto& world = mpi::Context::world();

  int32_t const rank = world.rank();

  /* argument checking and setting */
  auto const good =
      setup(argc, argv, &n, &energy, &niters, &px, &py, fname, world);

  if (not good) {
    return 0;
  }

  /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
  rx = rank % px;
  ry = rank / px;

  /* determine my four neighbors */
  north = (ry - 1) * px + rx;
  if (ry - 1 < 0) north = MPI_PROC_NULL;
  south = (ry + 1) * px + rx;
  if (ry + 1 >= py) south = MPI_PROC_NULL;
  west = ry * px + rx - 1;
  if (rx - 1 < 0) west = MPI_PROC_NULL;
  east = ry * px + rx + 1;
  if (rx + 1 >= px) east = MPI_PROC_NULL;

  FMPI_DBG(std::make_tuple(west, east, north, south));

  /* decompose the domain */
  bx   = n / px;  /* block size in x */
  by   = n / py;  /* block size in y */
  offx = rx * bx; /* offset in x */
  offy = ry * by; /* offset in y */

  /* divide blocks in x amongst threads */
  nthreads = omp_get_max_threads();
  Thx      = bx / nthreads;

  FMPI_DBG(nthreads);

  /* printf("nthreads: %d, Thx: %d\n", nthreads, Thx); */

  /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank,
   * ry,rx,west,east,north,south); */

  /* allocate working arrays & communication buffers */
  using value_t     = double;
  using mpi_alloc_t = fmpi::MpiAllocator<value_t>;
  using vector_t    = fmpi::SimpleVector<value_t, mpi_alloc_t>;

  std::size_t const mysize = (bx + 2) * (by + 2);
  auto              aold   = vector_t(mysize);
  auto              anew   = vector_t(mysize);

  /* initialize three heat sources */
  init_sources(
      bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

  /* create north-south datatype */
  MPI_Datatype north_south_type;
  MPI_Type_contiguous(Thx, MPI_DOUBLE, &north_south_type);
  MPI_Type_commit(&north_south_type);

  /* create east-west type */
  MPI_Datatype east_west_type;
  MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
  MPI_Type_commit(&east_west_type);

  t1 = MPI_Wtime(); /* take time */

  // setup dispatcher
  constexpr std::size_t n_types   = 2;
  constexpr std::size_t neighbors = 4;
  // std::size_t const                n_exchanges = nthreads;
  std::array<std::size_t, n_types> nslots{};
  nslots.fill(nthreads);

  for (iter = 0; iter < niters; ++iter) {
    /* refresh heat sources */
    for (i = 0; i < locnsources; ++i) {
      aold[ind(locsources[i][0], locsources[i][1])] +=
          energy; /* heat source */
    }

    /* reset the total heat */
    heat = 0.0;

    auto promise = fmpi::collective_promise{};
    auto future  = promise.get_future();
    auto schedule_state =
        std::make_unique<fmpi::ScheduleCtx>(nslots, std::move(promise));
    auto& dispatcher = fmpi::static_dispatcher_pool();

    int const tag_base = nthreads * neighbors;

    schedule_state->register_callback(
        fmpi::message_type::IRECV,
        [sptr = future.allocate_queue(tag_base)](
            const std::vector<fmpi::Message>& msgs) {
          for (auto&& msg : msgs) {
            sptr->push(msg);
          }
        });

    auto const hdl = dispatcher.submit(std::move(schedule_state));

    std::atomic_int32_t count_down = nthreads;

#pragma omp parallel private(i, j) reduction(+ : heat)
    {
      int thread_id = omp_get_thread_num();
      int xstart    = THX_START;
      int xend      = THX_END;

      const int     tag_start = tag_base - (thread_id * neighbors);
      constexpr int dim0      = 0;
      constexpr int dim1      = 1;

      if (north >= 0) {
        dispatcher.schedule(
            hdl,
            fmpi::message_type::ISENDRECV,
            fmpi::Message{
                &aold[ind(xstart, 1)] /* north */,
                1,
                north_south_type,
                static_cast<mpi::Rank>(north),
                tag_start - 2 * dim1 - 1,  // 0: 97, 24: 1
                &aold[ind(xstart, 0)] /* north */,
                1,
                north_south_type,
                static_cast<mpi::Rank>(north),
                tag_start - 2 * dim1,  // 0: 98, 24: 2
                world.mpiComm()});
      }
      /* exchange data with neighbors */
      if (south >= 0) {
        dispatcher.schedule(
            hdl,
            fmpi::message_type::ISENDRECV,
            fmpi::Message{
                &aold[ind(xstart, by)] /* south */,
                1,
                north_south_type,
                static_cast<mpi::Rank>(south),
                tag_start - 2 * dim1,  // 0: 98, 24: 2
                &aold[ind(xstart, by + 1)] /* south */,
                1,
                north_south_type,
                static_cast<mpi::Rank>(south),
                tag_start - 2 * dim1 - 1,  // 0: 97, 24: 1
                world.mpiComm()});
      }
      if ((west >= 0) && (xstart == 1)) {
        assert(thread_id == 0);
        // we receive from last thread in west
        int const tag_west = tag_base - ((nthreads - 1) * neighbors);

        dispatcher.schedule(
            hdl,
            fmpi::message_type::ISENDRECV,
            fmpi::Message{
                &aold[ind(1, 1)] /* west */,
                1,
                east_west_type,
                static_cast<mpi::Rank>(west),
                tag_west - 2 * dim0 - 1,  // 0: 3, 24: -
                &aold[ind(0, 1)] /* east */,
                1,
                east_west_type,
                static_cast<mpi::Rank>(west),
                tag_start - 2 * dim0,  // 0: 100, 24: -
                world.mpiComm()});
      }
      if ((east >= 0) && (xend == bx + 1)) {
        assert(thread_id == nthreads - 1);
        dispatcher.schedule(
            hdl,
            fmpi::message_type::ISENDRECV,
            fmpi::Message{
                &aold[ind(bx, 1)] /* east */,
                1,
                east_west_type,
                static_cast<mpi::Rank>(east),
                tag_base - 2 * dim0,  // 0: -, 24: 100
                &aold[ind(bx + 1, 1)] /* west */,
                1,
                east_west_type,
                static_cast<mpi::Rank>(east),
                tag_start - 2 * dim0 - 1,  // 0: -, 24: 3
                world.mpiComm()});
      }

      if (count_down.fetch_sub(1, std::memory_order_relaxed) == 1) {
        // the last thread schedules commits the collective comm
        dispatcher.commit(hdl);
      }

      // update inner grid points
      for (j = 2; j < by; ++j) {
        for (i = std::max(2, xstart); i < std::min(xend, bx); ++i) {
          anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                            (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                             aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) /
                                4.0 / 2.0;
          heat += anew[ind(i, j)];
        }
      }

#pragma omp single
      {
        future.wait();
        // implicit barrier here after single construct
      }

      // update outer grid points
      for (j = 1; j < by + 1; j += by - 1) {
        /* north, south -- two elements less per row (first and last) to
         avoid "double computation" in next loop! */
        for (i = std::max(2, xstart); i < std::min(xend, bx); ++i) {
          anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                            (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                             aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) /
                                4.0 / 2.0;
          heat += anew[ind(i, j)];
        }
      }

      if ((west >= 0) && (xstart == 1)) {
        // update outer grid points
        for (j = 1; j < by + 1; ++j) {
          for (i = 1; i < 2; ++i) {  // west -- full column
            anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                              (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                               aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) /
                                  4.0 / 2.0;
            heat += anew[ind(i, j)];
          }
        }
      }

      if ((east >= 0) && (xend == bx + 1)) {
        for (j = 1; j < by + 1; ++j) {
          for (i = bx; i < bx + 1; ++i) {  // east -- full column
            anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                              (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                               aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) /
                                  4.0 / 2.0;
            heat += anew[ind(i, j)];
          }
        }
      }

    } /* end parallel region */

    /* swap working arrays */
    std::swap(aold, anew);

    /* optional - print image */
    if (iter == niters - 1) {
      printarr_par(
          iter,
          anew.data(),
          n,
          px,
          py,
          rx,
          ry,
          bx,
          by,
          offx,
          offy,
          fname.c_str(),
          world.mpiComm());
    }
  }

  t2 = MPI_Wtime();

  MPI_Type_free(&east_west_type);
  MPI_Type_free(&north_south_type);

  /* get final heat in the system */
  MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, world.mpiComm());
  if (!rank) printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);
  return 0;
}

bool setup(
    int                 argc,
    char**              argv,
    int*                n_ptr,
    int*                energy_ptr,
    int*                niters_ptr,
    int*                px_ptr,
    int*                py_ptr,
    std::string&        output,
    mpi::Context const& ctx) {
  bool good = false;

  tlx::CmdlineParser cp;

  // add description and author
  cp.set_description("Stencil Example with FMPI.");
  cp.set_author("Roger Kowalewski <roger.kowaleski@nm.ifi.lmu.de>");

  cp.add_param_int("length", *n_ptr, "Size of a dimension in global grid.");
  cp.add_param_int(
      "energy", *energy_ptr, "energy to be injected per iteration");
  cp.add_param_int("iterations", *niters_ptr, "Number of iterations.");
  cp.add_param_int("px", *px_ptr, "Number of processors in x-dimension");
  cp.add_param_int("py", *py_ptr, "Number of processors in y-dimension");
  cp.add_param_string("output", output, "Output file");

  auto const me = ctx.rank();

  if (me == 0) {
    good = cp.process(argc, argv, std::cout);
  } else {
    rtlx::onullstream os;
    good = cp.process(argc, argv, os);
  }

  if (not good) {
    return false;
  }

  if (*px_ptr * (*py_ptr) != static_cast<int>(ctx.size()))
    ctx.abort(1);                          /* abort if px or py are wrong */
  if (*n_ptr % *py_ptr != 0) ctx.abort(2); /* abort px needs to divide n */
  if (*n_ptr % *px_ptr != 0) ctx.abort(3); /* abort py needs to divide n */

  return good;
}

void init_sources(
    int       bx,
    int       by,
    int       offx,
    int       offy,
    int       n,
    const int nsources,
    int       sources[][2],
    int*      locnsources_ptr,
    int       locsources[][2]) {
  int i, locnsources = 0;

  sources[0][0] = n / 2;
  sources[0][1] = n / 2;
  sources[1][0] = n / 3;
  sources[1][1] = n / 3;
  sources[2][0] = n * 4 / 5;
  sources[2][1] = n * 8 / 9;

  for (i = 0; i < nsources;
       ++i) { /* determine which sources are in my patch */
    int locx = sources[i][0] - offx;
    int locy = sources[i][1] - offy;
    if (locx >= 0 && locx < bx && locy >= 0 && locy < by) {
      locsources[locnsources][0] = locx + 1; /* offset by halo zone */
      locsources[locnsources][1] = locy + 1; /* offset by halo zone */
      locnsources++;
    }
  }

  (*locnsources_ptr) = locnsources;
}
