add_library(mpi_fusion mpi/Environment.cc Schedule.cc)

target_include_directories(mpi_fusion PUBLIC ../include)

target_link_libraries(mpi_fusion PUBLIC MPI::MPI_CXX)
target_link_libraries(mpi_fusion PRIVATE fusion_debug tlx)
