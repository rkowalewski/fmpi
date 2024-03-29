option(ENABLE_CLANG_TIDY "generate a Clang-Tidy target to lint project sources" OFF)

if(CMAKE_VERSION VERSION_GREATER 3.6 AND ENABLE_CLANG_TIDY MATCHES "ON")
    option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)

    # Add clang-tidy if available
    find_program(
        CLANG_TIDY_BIN
        NAMES
            clang-tidy
            clang-tidy-7
            clang-tidy-6
            clang-tidy-5
        DOC "Path to clang-tidy executable"
    )


    if(NOT CLANG_TIDY_BIN)
        message(FATAL_ERROR "unable to locate clang-tidy")
    endif()

    find_program(
        RUN_CLANG_TIDY_BIN
        NAMES
            run-clang-tidy
            run-clang-tidy-7
    )

    if(NOT RUN_CLANG_TIDY_BIN)
        message(FATAL_ERROR "unable to locate run-clang-tidy-5.0.py")
    endif()

    execute_process(
        COMMAND bash "-c" "git -C ${CMAKE_SOURCE_DIR} ls-files | grep \"\\.cc$\" | grep -v \".*external/\""
        OUTPUT_VARIABLE TRACKED_FILES)


    string(REPLACE "\n" ";" TRACKED_FILES ${TRACKED_FILES})

    list(APPEND RUN_CLANG_TIDY_BIN_ARGS
        -clang-tidy-binary=${CLANG_TIDY_BIN}
        "\"-header-filter=.*${CMAKE_SOURCE_DIR}/(include|libs|src|benchmark)/.*\""
        -export-fixes fixes.yaml
        -format
        -style file)

    add_custom_target(
        tidy
        COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS} ${TRACKED_FILES}
        COMMENT "running clang tidy"
        )
endif()
