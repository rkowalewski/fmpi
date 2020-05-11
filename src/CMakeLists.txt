add_library(fmpi
    concurrency/CacheLocality.cc
    detail/Assert.cc
    memory/BlockAllocator.cc
    memory/ThreadAllocator.cc
    mpi/Algorithm.cc
    mpi/Environment.cc
    mpi/Rank.cc
    mpi/Request.cc
    util/Trace.cc
    common/Porting.cc
    Schedule.cc
    Pinning.cc
    Dispatcher.cc
    )


set(MI_OVERRIDE OFF)

target_include_directories(fmpi PUBLIC ../include ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(fmpi PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Boost::boost
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


if (NOT MSVC)
    target_compile_options(fmpi PUBLIC -Wall -Wextra)
endif()

if (PEDANTIC_SUPPORTED)
    target_compile_options(fmpi PUBLIC -pedantic)
endif()

if (CAST_QUAL_SUPPORTED)
    target_compile_options(fmpi PUBLIC -Wcast-qual)
endif()
if (INIT_SELF_SUPPORTED)
    target_compile_options(fmpi PUBLIC -Winit-self)
endif()
if (OVERLOADED_VIRTUAL_SUPPORTED)
    target_compile_options(fmpi PUBLIC -Woverloaded-virtual)
endif()

# configure config file
configure_file("Config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/config_impl.hpp")

#if (REDUNDANT_DECLS_SUPPORTED)
#    target_compile_options(fmpi PUBLIC -Wredundant-decls)
#endif()

