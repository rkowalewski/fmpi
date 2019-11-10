add_library(fmpi mpi/Environment.cc mpi/Algorithm.cc Schedule.cc)

target_include_directories(fmpi PUBLIC ../include)

target_link_libraries(fmpi PUBLIC MPI::MPI_CXX dbg_macro)
target_link_libraries(fmpi PRIVATE rtlx tlx)

if (UNIX)
    execute_process(COMMAND getconf LEVEL3_CACHE_SIZE
        OUTPUT_VARIABLE _l3CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND getconf LEVEL2_CACHE_SIZE
        OUTPUT_VARIABLE _l2CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_compile_definitions (fmpi PRIVATE FMPI_CACHELEVEL3_SIZE=${_l3CacheSize})
    target_compile_definitions (fmpi PRIVATE FMPI_CACHELEVEL2_SIZE=${_l2CacheSize})
endif()

