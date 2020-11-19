add_library(fmpi
    alltoall.cc
    Dispatcher.cc
    Pinning.cc
    Message.cc
    common/Porting.cc
    concurrency/CacheLocality.cc
    concurrency/Future.cc
    detail/Assert.cc
    memory/BlockAllocator.cc
    memory/ThreadAllocator.cc
    mpi/Algorithm.cc
    mpi/Environment.cc
    mpi/Rank.cc
    mpi/Request.cc
    neighbor_alltoall.cc
    # topo/BinaryTree.cc
    topo/Tree.cc
    util/Trace.cc
    )

set(MI_OVERRIDE OFF)

target_include_directories(fmpi PUBLIC ../include ${CMAKE_CURRENT_BINARY_DIR})
# target_include_directories(fmpi PUBLIC ${JUNCTION_ALL_INCLUDE_DIRS})

target_link_libraries(fmpi PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Boost::boost
    #   ${JUNCTION_ALL_LIBRARIES}
    dbg_macro
    GSL
    tlx
    rtlx)

target_compile_options(fmpi PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(fmpi PRIVATE snmalloc_lib)

execute_process(COMMAND getconf LEVEL3_CACHE_SIZE
    OUTPUT_VARIABLE FMPI_CACHELEVEL3_SIZE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND getconf LEVEL2_CACHE_SIZE
    OUTPUT_VARIABLE FMPI_CACHELEVEL2_SIZE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND getconf LEVEL1_DCACHE_SIZE
    OUTPUT_VARIABLE FMPI_CACHELEVEL1_SIZE OUTPUT_STRIP_TRAILING_WHITESPACE)

if ("${CMAKE_BUILD_TYPE}" MATCHES ".*(D|d)ebug$")
    target_compile_definitions (fmpi PRIVATE DBG_MACRO_NO_WARNING)
    set(FMPI_DEBUG_ASSERT ON CACHE BOOL "" FORCE)
else()
    set(FMPI_DEBUG_ASSERT OFF CACHE BOOL "" FORCE)
endif()

set_project_warnings(${PROJECT_NAME})

# configure config file
configure_file("Config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/config_impl.hpp")
