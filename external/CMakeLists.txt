add_subdirectory(tlx)
add_subdirectory(libmorton)
add_subdirectory(dbg-macro)

add_subdirectory(google/benchmark)

add_library(MPISynchronizedBarrier MPISynchronizedBarrier/MPISynchronizedBarrier.cc)

target_include_directories(MPISynchronizedBarrier PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/MPISynchronizedBarrier/
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/MPISynchronizedBarrier/)

target_link_libraries(MPISynchronizedBarrier PRIVATE MPI::MPI_CXX)


#add_library(BenchmarkTime INTERFACE)
#target_include_directories(BenchmarkTime INTERFACE Benchmark)
