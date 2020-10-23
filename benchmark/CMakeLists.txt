add_executable(twoSidedAlgorithms apps/TwosidedAlgorithms.cc src/Params.cc)

set_target_properties(twoSidedAlgorithms PROPERTIES DEBUG_POSTFIX ".${CMAKE_DEBUG_POSTFIX}")

add_dependencies(twoSidedAlgorithms GitVersion)
target_include_directories(twoSidedAlgorithms PRIVATE include apps src)

# Make sure you link your targets with this command. It can also link libraries and
# even flags, so linking a target that does not exist will not give a configure-time error.
target_link_libraries(twoSidedAlgorithms MPISynchronizedBarrier tlx Morton rtlx fmpi)

add_custom_target(GitVersion
    COMMAND ${CMAKE_COMMAND}
    -D INPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h.in
    -D OUTPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h
    -P ${CMAKE_SOURCE_DIR}/cmake/generate_version.cmake
)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")


add_executable(osu_latency apps/osu_latency.cc)
set_target_properties(osu_latency PROPERTIES DEBUG_POSTFIX ".${CMAKE_DEBUG_POSTFIX}")
target_link_libraries(osu_latency fmpi)
