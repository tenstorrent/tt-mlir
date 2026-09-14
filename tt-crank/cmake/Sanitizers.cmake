# tt_crank_sanitizers: INTERFACE library carrying sanitizer compile + link flags.
# Link PRIVATE from first-party targets only — never to third-party (tt-mlir, gtest),
# whose unsanitized objects share the same executable. See the plan's "Risks" section
# for the partial-coverage caveat.

add_library(tt_crank_sanitizers INTERFACE)

option(TT_CRANK_ENABLE_ASAN  "Enable AddressSanitizer for tt-crank code"          OFF)
option(TT_CRANK_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for tt-crank code" OFF)
option(TT_CRANK_ENABLE_TSAN  "Enable ThreadSanitizer (exclusive with ASan/UBSan)"   OFF)

if(TT_CRANK_ENABLE_TSAN AND (TT_CRANK_ENABLE_ASAN OR TT_CRANK_ENABLE_UBSAN))
    message(FATAL_ERROR
        "TT_CRANK_ENABLE_TSAN is mutually exclusive with TT_CRANK_ENABLE_ASAN / TT_CRANK_ENABLE_UBSAN.")
endif()

set(_TT_CRANK_SAN_FLAGS "")
if(TT_CRANK_ENABLE_ASAN)
    list(APPEND _TT_CRANK_SAN_FLAGS -fsanitize=address -fno-omit-frame-pointer)
endif()
if(TT_CRANK_ENABLE_UBSAN)
    list(APPEND _TT_CRANK_SAN_FLAGS -fsanitize=undefined)
endif()
if(TT_CRANK_ENABLE_TSAN)
    list(APPEND _TT_CRANK_SAN_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
endif()

if(_TT_CRANK_SAN_FLAGS)
    target_compile_options(tt_crank_sanitizers INTERFACE ${_TT_CRANK_SAN_FLAGS})
    target_link_options(tt_crank_sanitizers    INTERFACE ${_TT_CRANK_SAN_FLAGS})
endif()
