add_subdirectory(tlx)
add_subdirectory(libmorton)
add_subdirectory(dbg-macro)
add_subdirectory(GSL)
add_subdirectory(google/benchmark)

# SNMALLOC
set(SNMALLOC_ONLY_HEADER_LIBRARY ON)
add_subdirectory(snmalloc EXCLUDE_FROM_ALL)

# JUNCTION
#add_subdirectory(junction)
#set(JUNCTION_ALL_INCLUDE_DIRS ${JUNCTION_ALL_INCLUDE_DIRS} PARENT_SCOPE)

# MPI Synchronized Barrier
add_library(MPISynchronizedBarrier MPISynchronizedBarrier/MPISynchronizedBarrier.cc)
target_include_directories(MPISynchronizedBarrier PUBLIC MPISynchronizedBarrier)
target_link_libraries(MPISynchronizedBarrier PRIVATE MPI::MPI_CXX)

