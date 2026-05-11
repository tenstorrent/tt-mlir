# Wire up ccache as the compiler launcher if it's available on PATH.

option(TT_KURBLA_USE_CCACHE "Use ccache as compiler launcher if available" ON)

if(TT_KURBLA_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "ccache found at ${CCACHE_PROGRAM}; using as compiler launcher")
        set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()
