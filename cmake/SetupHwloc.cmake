if (NOT TARGET fmpi::hwloc)

  find_package(Hwloc)
  if(NOT HWLOC_FOUND)
    message("Hwloc could not be found, please specify HWLOC_ROOT to point to the correct location")
  endif()

  add_library(fmpi::hwloc INTERFACE IMPORTED)
  # System has been removed when passing at set_property for cmake < 3.11
  # instead of target_include_directories
  set_property(TARGET fmpi::hwloc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HWLOC_INCLUDE_DIR})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET fmpi::hwloc PROPERTY INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARIES})
  else()
    target_link_libraries(fmpi::hwloc INTERFACE ${HWLOC_LIBRARIES})
  endif()

endif()
