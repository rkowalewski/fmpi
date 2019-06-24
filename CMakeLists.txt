# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(AlltoAllAlgos VERSION 1.0 LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Find packages go here.
option(MPI_CXX_SKIP_MPICXX "" ON)
find_package(MPI REQUIRED)

# For supporting CMake < 3.9:
if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)

    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

find_package(OpenMP)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)

endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-pedantic PEDANTIC_SUPPORTED)

if (NOT MSVC)
	add_compile_options(-Wall -Wextra -Werror)

    if (PEDANTIC_SUPPORTED)
	    add_compile_options(-Werror)
    endif()
endif()


message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# You should usually split this into folders, but this is a simple example

add_subdirectory(external/tlx)

# This is a "default" library, and will match the *** variable setting.
# Other common choices are STATIC, SHARED, and MODULE
# Including header files here helps IDEs but is not required.
# Output libname matches target name, with the usual extensions on your system

add_library(MPISynchronizedBarrier external/MPISynchronizedBarrier/MPISynchronizedBarrier.cc)
target_include_directories(MPISynchronizedBarrier PUBLIC external/MPISynchronizedBarrier)
target_link_libraries(MPISynchronizedBarrier PRIVATE MPI::MPI_CXX)

# Link each target with other targets or add options, etc.

# Adding something we can run - Output name matches target name
add_executable(MpiAlltoAllBench src/MpiAlltoAllBench.cc src/MpiAlltoAllBench.h src/Trace.cc)
add_dependencies(MpiAlltoAllBench GitVersion)
target_include_directories(MpiAlltoAllBench PUBLIC include external/tlx)
target_include_directories(MpiAlltoAllBench PUBLIC src)

# Make sure you link your targets with this command. It can also link libraries and
# even flags, so linking a target that does not exist will not give a configure-time error.
target_link_libraries(MpiAlltoAllBench PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX MPISynchronizedBarrier)
# link tlx
target_compile_options(MpiAlltoAllBench PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(MpiAlltoAllBench PUBLIC tlx)


add_executable(factor_serial src/factor_serial.cc)

if(CMAKE_VERSION VERSION_GREATER 3.6 AND ENABLE_CLANG_TIDY)
    # Add clang-tidy if available
    option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)
    find_program(
        CLANG_TIDY_EXE
        NAMES "clang-tidy"
        DOC "Path to clang-tidy executable"
    )

    if(CLANG_TIDY_EXE)
        if(CLANG_TIDY_FIX)
            set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "-fix")
        else()
            set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
        endif()
    endif()

    # if(RUN_CLANG_TIDY_BIN STREQUAL "RUN_CLANG_TIDY_BIN-NOTFOUND")
    #    message(FATAL_ERROR "unable to locate run-clang-tidy-5.0.py")
   # endif()

    # list(APPEND RUN_CLANG_TIDY_BIN_ARGS
    #     -clang-tidy-binary ${CLANG_TIDY_BIN}
    #     "\"-header-filter=.*\\b(src|test|examples)\\b\\/(?!lib).*\""     #Only run clang tidy on src, test, examples and skip 3rd party libraries
    #     -checks=clan*,cert*,misc*,perf*,cppc*,read*,mode*,-cert-err58-cpp,-misc-noexcept-move-constructor,-cppcoreguidelines-*
    #     )

    #add_custom_target(
    #    tidy
    #    COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
    #    COMMENT "running clang tidy"
    #    )
endif()

if (ENABLE_LOGGING)
    target_compile_definitions (MpiAlltoAllBench PRIVATE A2A_ENABLE_LOGGING)
endif()

add_custom_target(GitVersion
    COMMAND ${CMAKE_COMMAND}
    -D INPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h.in
    -D OUTPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_version.cmake
)

