add_library(fusion_timer timer/Trace.cc)

target_include_directories(fusion_timer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(fusion_timer PRIVATE fusion_debug)


add_library(fusion_debug INTERFACE)
target_include_directories(fusion_debug INTERFACE debug)


#target_compile_options(dsss_mpi
#  PRIVATE
#  ${DSSS_FLAGS}
#  $<$<CONFIG:Debug>:${DSSS_DEBUG_FLAGS}>
#  $<$<CONFIG:Release>:${DSSS_RELEASE_FLAGS}>)
