# Run clang-tidy during compilation of first-party code when opted in.
# Third-party targets (gtest, tt-mlir) explicitly clear CXX_CLANG_TIDY so this
# global default doesn't apply to them.

option(TT_CRANK_ENABLE_CLANG_TIDY "Run clang-tidy during compilation of first-party code" OFF)

if(TT_CRANK_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_PROGRAM
        NAMES clang-tidy
        HINTS "$ENV{TTMLIR_TOOLCHAIN_DIR}/venv/bin"
    )
    if(CLANG_TIDY_PROGRAM)
        message(STATUS "clang-tidy found at ${CLANG_TIDY_PROGRAM}; will run during build")
        # `--extra-arg-before=-Wno-unknown-warning-option` lets clang-tidy silently
        # ignore GCC-only -W flags (e.g. -Wduplicated-cond) that get passed when we
        # compile our code with gcc. Without this, clang-tidy errors out on them.
        set(CMAKE_CXX_CLANG_TIDY
            "${CLANG_TIDY_PROGRAM};--use-color;--extra-arg-before=-Wno-unknown-warning-option")
    else()
        message(WARNING "TT_CRANK_ENABLE_CLANG_TIDY=ON but clang-tidy was not found; skipping")
    endif()
endif()
