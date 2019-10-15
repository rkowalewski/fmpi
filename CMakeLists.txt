# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(MpiFusion VERSION 1.0 LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Find packages go here.
option(MPI_CXX_SKIP_MPICXX "" ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-pedantic PEDANTIC_SUPPORTED)

if (NOT MSVC)
	add_compile_options(-Wall -Wextra -Werror)

    if (PEDANTIC_SUPPORTED)
	    add_compile_options(-Werror)
    endif()
endif()

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/FindMPI.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindOpenMP.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/ClangTidy.cmake)

add_subdirectory(external)
add_subdirectory(libs)
add_subdirectory(src)

add_subdirectory(apps)
add_subdirectory(benchmark)


