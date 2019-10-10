if(CMAKE_VERSION VERSION_GREATER 3.6 AND ENABLE_CLANG_TIDY)
    # Add clang-tidy if available
    option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)
    find_program(
        CLANG_TIDY_BIN
        NAMES
            clang-tidy-7
            clang-tidy-6
            clang-tidy-5
            clang-tidy
        DOC "Path to clang-tidy executable"
    )

    if(NOT CLANG_TIDY_BIN)
        message(FATAL_ERROR "unable to locate clang-tidy")
    endif()

    if(RUN_CLANG_TIDY_BIN STREQUAL "RUN_CLANG_TIDY_BIN-NOTFOUND")
        message(FATAL_ERROR "unable to locate run-clang-tidy-5.0.py")
    endif()

    list(APPEND RUN_CLANG_TIDY_BIN_ARGS
        -clang-tidy-binary ${CLANG_TIDY_BIN}
        "\"-header-filter=${CMAKE_SOURCE_DIR}/include/*\""
        -checks="*,-hicpp-no-array-decay,-fuchsia*,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-clang-analyzer-core.NonNull*,-clang-analyzer-core.NullDereference,-clang-analyzer-core.uninitialized.Branch"
        ${CMAKE_SOURCE_DIR}/src/*
        )

    if (CLANG_TIDY_FIX)
        list(APPEND RUN_CLANG_TIDY_BIN_ARGS
            "-fix")
    endif()

    add_custom_target(
        tidy
        COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
        COMMENT "running clang tidy"
        )
endif()