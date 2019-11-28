# Make an automatic library - will be static or dynamic based on user setting
add_library(rtlx Trace.cc)

# We need this directory, and users of our library will need it too
target_include_directories(rtlx PUBLIC ../include)

if (RTLX_ENABLE_TRACE MATCHES "ON")
    list(APPEND rtlx_defines RTLX_ENABLE_TRACE)
endif()

target_compile_definitions (rtlx PRIVATE ${rtlx_defines})

# All users of this library will need at least C++11
target_compile_features(rtlx PUBLIC cxx_std_11)
