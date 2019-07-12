#include <mpi.h>
#include <cstdio>

int main(int argc, char** argv)
{
  int  len;
  char mpi_lib_ver[MPI_MAX_LIBRARY_VERSION_STRING];

  MPI_Init(NULL, NULL);
  MPI_Get_library_version(mpi_lib_ver, &len);
  printf("MPI library version: %s\n", mpi_lib_ver);

  return 0;
}
