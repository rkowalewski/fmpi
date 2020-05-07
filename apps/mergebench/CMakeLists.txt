add_executable(parallel_recursive src/recursive.cc src/benchmark.cc)
add_executable(parallel src/parallel.cc src/benchmark.cc)

target_link_libraries(parallel_recursive PRIVATE
    fmpi
    benchmark
)

target_link_libraries(parallel PRIVATE
    fmpi
    benchmark
)

