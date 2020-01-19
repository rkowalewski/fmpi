# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(fmpi VERSION 1.0 LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

if (NOT CMAKE_BUILD_TYPE)
    if ("${CMAKE_BINARY_DIR}" MATCHES ".*(R|r)elease$")
        message(STATUS "No build type selected, default to: Release")
        set(CMAKE_BUILD_TYPE "Release")
    else()
        message(STATUS "No build type selected, default to: Debug")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
else()
    message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
endif()

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

STRING(STRIP "${ICC_GXX}" ICC_GXX)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel" AND NOT "${ICC_GXX}" STREQUAL "")
    message(STATUS "Using GCC: ${ICC_GXX}")
    string(APPEND CMAKE_CXX_FLAGS "-gxx-name=${ICC_GXX}")
else()
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Options
option(MPI_CXX_SKIP_MPICXX "" ON)
option(WITH_HWLOC           "Enable hwloc support" OFF)

include(${CMAKE_SOURCE_DIR}/cmake/FindMPI.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindOpenMP.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/ClangTidy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/SetupBoost.cmake)

if (WITH_HWLOC)
    include(${CMAKE_SOURCE_DIR}/cmake/SetupHwloc.cmake)
endif()


set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable")

add_subdirectory(external)
add_subdirectory(libs)
add_subdirectory(src)

add_subdirectory(apps)
add_subdirectory(benchmark)
