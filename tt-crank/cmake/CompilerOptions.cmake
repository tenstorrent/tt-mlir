# tt_kurbla_warnings: an INTERFACE library carrying our standard warning flags.
# Link PRIVATE from first-party targets only — third-party (gtest, tt-mlir, ...)
# is excluded so we don't fail on warnings we can't control.

add_library(tt_kurbla_warnings INTERFACE)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

option(TT_KURBLA_WARNINGS_AS_ERRORS "Treat warnings as errors for tt-kurbla code" OFF)

set(_TT_KURBLA_WARNINGS_GCC_CLANG
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Werror=return-type
)

set(_TT_KURBLA_WARNINGS_GCC_ONLY
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
    -Wsuggest-override
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(tt_kurbla_warnings INTERFACE ${_TT_KURBLA_WARNINGS_GCC_CLANG})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(tt_kurbla_warnings INTERFACE ${_TT_KURBLA_WARNINGS_GCC_ONLY})
    endif()

    if(TT_KURBLA_WARNINGS_AS_ERRORS)
        target_compile_options(tt_kurbla_warnings INTERFACE -Werror)
    endif()
endif()
