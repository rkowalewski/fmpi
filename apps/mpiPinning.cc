#include <mpi.h>

#include <iostream>
#include <sstream>

#include <fmpi/mpi/Environment.h>

#include <rtlx/Assert.h>
#include <rtlx/ScopedLambda.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  auto finalizer = rtlx::scope_exit([]() { MPI_Finalize(); });

  mpi::Context worldCtx{MPI_COMM_WORLD};

  auto me = worldCtx.rank();
  auto nr = worldCtx.size();

  char _hostname[MPI_MAX_PROCESSOR_NAME];
  int  resultlen;

  RTLX_ASSERT_RETURNS(
      MPI_Get_processor_name(_hostname, &resultlen), MPI_SUCCESS);

  std::string hostname(_hostname);

  std::ostringstream os;
  os << "I am " << hostname << " (rank " << me << " of " << nr << ")\n";

  std::cout << os.str();

  return 0;
}
