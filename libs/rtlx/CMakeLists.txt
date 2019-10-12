# Works with 3.11 and tested through 3.15
cmake_minimum_required(VERSION 3.0)

# Project name and a few useful settings. Other commands can pick up the results
project(rtlx
    VERSION 0.1
    DESCRIPTION "Use C++ Stuff"
    LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()


# The compiled library code is here
add_subdirectory(src)
