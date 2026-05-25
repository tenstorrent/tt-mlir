# Core utilities: assert/cast headers + spdlog logging.
# All targets (engine, torch, tests) should link with this to get assert/cast/logger.
add_library(tt_kurbla_common SHARED
    assert.cpp
    config.cpp
    misc.cpp
)
add_library(tt_kurbla::common ALIAS tt_kurbla_common)

target_include_directories(tt_kurbla_common
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/common>
)

target_compile_features(tt_kurbla_common PUBLIC cxx_std_20)

# Controls which spdlog call sites are compiled into assert.cpp.
target_compile_definitions(tt_kurbla_common PUBLIC
    $<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
)

target_link_libraries(tt_kurbla_common
    PUBLIC
        tt-logger::tt-logger
    PRIVATE
        tt_kurbla_warnings
        tt_kurbla_sanitizers
)
