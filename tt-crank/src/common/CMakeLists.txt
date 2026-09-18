# Core utilities: assert/cast headers + spdlog logging.
# All targets (engine, torch, tests) should link with this to get assert/cast/logger.
add_library(tt_crank_common SHARED
    assert.cpp
    config.cpp
    misc.cpp
)
add_library(tt_crank::common ALIAS tt_crank_common)

target_include_directories(tt_crank_common
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/common>
)

target_compile_features(tt_crank_common PUBLIC cxx_std_20)

# Controls which spdlog call sites are compiled into assert.cpp.
target_compile_definitions(tt_crank_common PUBLIC
    $<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
)

target_link_libraries(tt_crank_common
    PUBLIC
        tt-logger::tt-logger
    PRIVATE
        tt_crank_warnings
        tt_crank_sanitizers
)

# Wheel: goes in tt_crank/lib with the other native libs. Only the installed copy
# gets this RPATH; build-tree binaries keep CMake's build RPATH.
set_target_properties(tt_crank_common PROPERTIES INSTALL_RPATH "$ORIGIN")
