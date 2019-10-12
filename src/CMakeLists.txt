add_library(fmpi mpi/Environment.cc Schedule.cc)

target_include_directories(fmpi PUBLIC ../include)

target_link_libraries(fmpi PUBLIC MPI::MPI_CXX)
target_link_libraries(fmpi PRIVATE rtlx tlx)
