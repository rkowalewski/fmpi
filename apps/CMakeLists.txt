add_executable(factor_serial factor_serial.cc)

add_executable(mpiPinning mpiPinning.cc)
target_link_libraries(mpiPinning PRIVATE
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    rtlx
    fmpi
    benchmark
)

add_executable(mergebench mergebench.cc)
target_link_libraries(mergebench PRIVATE
    fmpi
    benchmark
    OpenMP::OpenMP_CXX
    tlx
)

