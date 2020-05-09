# Works with 3.11 and tested through 3.15
cmake_minimum_required(VERSION 3.0)

# Project name and a few useful settings. Other commands can pick up the results
project(rtlx
    VERSION 0.1
    DESCRIPTION "C++ Stuff"
    LANGUAGES CXX)

# Make an automatic library - will be static or dynamic based on user setting
add_library(rtlx INTERFACE)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_EXTENSIONS OFF)

endif()

# We need this directory, and users of our library will need it too
target_include_directories(rtlx INTERFACE include)

# All users of this library will need at least C++11
target_compile_features(rtlx INTERFACE cxx_std_11)

