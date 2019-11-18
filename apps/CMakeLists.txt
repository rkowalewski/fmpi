add_executable(factor_serial factor_serial.cc)

add_executable(mpiPinning mpiPinning.cc)
target_link_libraries(mpiPinning PRIVATE OpenMP::OpenMP_CXX
    MPI::MPI_CXX rtlx fmpi)
