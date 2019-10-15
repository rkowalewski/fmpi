add_executable(mpiFusionBench src/MpiAlltoAllBench.cc)

set_target_properties(mpiFusionBench PROPERTIES DEBUG_POSTFIX ".${CMAKE_DEBUG_POSTFIX}")

add_dependencies(mpiFusionBench GitVersion)
target_include_directories(mpiFusionBench PRIVATE include src)

# Make sure you link your targets with this command. It can also link libraries and
# even flags, so linking a target that does not exist will not give a configure-time error.
target_link_libraries(mpiFusionBench PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX
    MPISynchronizedBarrier tlx Morton rtlx fmpi)
# link tlx
target_compile_options(mpiFusionBench PRIVATE ${OpenMP_CXX_FLAGS})

if (UNIX)
    execute_process(COMMAND getconf LEVEL3_CACHE_SIZE
        OUTPUT_VARIABLE _l3CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND getconf LEVEL2_CACHE_SIZE
        OUTPUT_VARIABLE _l2CacheSize OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_compile_definitions (mpiFusionBench PRIVATE A2A_CACHELEVEL3_SIZE=${_l3CacheSize})
    target_compile_definitions (mpiFusionBench PRIVATE A2A_CACHELEVEL2_SIZE=${_l2CacheSize})
endif()

if (ENABLE_LOGGING)
    target_compile_definitions (mpiFusionBench PRIVATE A2A_ENABLE_LOGGING)
endif()


add_custom_target(GitVersion
    COMMAND ${CMAKE_COMMAND}
    -D INPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h.in
    -D OUTPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h
    -P ${CMAKE_SOURCE_DIR}/cmake/generate_version.cmake
)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

