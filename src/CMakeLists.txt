add_library(fmpi
    concurrency/CacheLocality.cc
    memory/BlockAllocator.cc
    mpi/Algorithm.cc
    mpi/Environment.cc
    mpi/Rank.cc
    mpi/Request.cc
    common/Porting.cc
    Schedule.cc
    Config.cc
    Dispatcher.cc
    )

target_include_directories(fmpi PUBLIC ../include)

target_link_libraries(fmpi PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX Boost::boost dbg_macro GSL)
target_compile_options(fmpi PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(fmpi PRIVATE rtlx tlx)

if(HWLOC_FOUND)
    target_link_libraries(fmpi PRIVATE fmpi::hwloc)
    target_compile_definitions (fmpi PRIVATE FMPI_USEHWLOC)
endif()

if (UNIX)
    execute_process(COMMAND getconf LEVEL3_CACHE_SIZE
        OUTPUT_VARIABLE _l3CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND getconf LEVEL2_CACHE_SIZE
        OUTPUT_VARIABLE _l2CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_compile_definitions (fmpi PRIVATE FMPI_CACHELEVEL3_SIZE=${_l3CacheSize})
    target_compile_definitions (fmpi PRIVATE FMPI_CACHELEVEL2_SIZE=${_l2CacheSize})
endif()

if ("${CMAKE_BUILD_TYPE}" MATCHES ".*(D|d)ebug$")
    target_compile_definitions (fmpi PRIVATE DBG_MACRO_NO_WARNING)
endif()


