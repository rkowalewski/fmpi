add_executable(factor_serial factor_serial.cc)
target_link_libraries(factor_serial PRIVATE
    fmpi
)

add_executable(mpiPinning mpiPinning.cc)
target_link_libraries(mpiPinning PRIVATE
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    rtlx
    fmpi
)

add_executable(mpiSpsc mpi_spsc.cc)
target_link_libraries(mpiSpsc PRIVATE
    fmpi
    OpenMP::OpenMP_CXX
    Boost::boost
    tlx
    rtlx
)

add_subdirectory(mergebench)

