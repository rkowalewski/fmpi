add_subdirectory(tlx)
add_subdirectory(libmorton)
add_subdirectory(dbg-macro)
# add_subdirectory(SPSCQueue INTERFACE)

add_subdirectory(google/benchmark)



add_library(MPISynchronizedBarrier MPISynchronizedBarrier/MPISynchronizedBarrier.cc)
target_include_directories(MPISynchronizedBarrier PUBLIC MPISynchronizedBarrier)
target_link_libraries(MPISynchronizedBarrier PRIVATE MPI::MPI_CXX)

