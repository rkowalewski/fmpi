FIND_PACKAGE(Git)
IF(GIT_FOUND)
  EXECUTE_PROCESS(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE "A2A_GIT_COMMIT"
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  MESSAGE( STATUS "Git version: ${A2A_GIT_COMMIT}" )
ELSE(GIT_FOUND)
  SET(A2A_GIT_COMMIT 0)
ENDIF(GIT_FOUND)

configure_file(${INPUT_FILE} ${OUTPUT_FILE} @ONLY)

