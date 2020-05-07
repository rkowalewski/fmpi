add_executable(parallel_recursive recursive.cc benchmark.cc)
add_executable(parallel parallel.cc benchmark.cc)

target_link_libraries(parallel_recursive PRIVATE
    fmpi
    benchmark
)

target_link_libraries(parallel PRIVATE
    fmpi
    benchmark
)

