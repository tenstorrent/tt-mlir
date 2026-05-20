include(GenerateExportHeader)

configure_file(
    version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    @ONLY
)

add_library(tt_kurbla SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    engine/compile.cpp
    engine/device.cpp
    engine/execution_payload.cpp
    # Static-init env shim runtime. Linked into the library so every consumer (tests,
    # Python wheel, future bindings) picks up the ttsim routing when the env var is set.
    # TT_KURBLA_SIM_DIR / TT_KURBLA_TT_METAL_HOME defs are wired in cmake/TTsim.cmake.
    engine/sim_env.cpp
)
add_library(tt_kurbla::tt_kurbla ALIAS tt_kurbla)

# Hide every symbol by default; only declarations marked TT_KURBLA_API end up in
# the .so's dynamic symbol table. Without this, every namespace-scope symbol
# (including unnamed-namespace helpers and tt-mlir template instantiations
# reachable from our headers) would leak into the public ABI on Linux.
set_target_properties(tt_kurbla PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Generates tt_kurbla_export.hpp defining TT_KURBLA_API (plus deprecation
# helpers). The macro expands to the export attribute when building tt_kurbla
# itself (tt_kurbla_EXPORTS is auto-defined by CMake for shared targets) and to
# the import attribute when consumers compile against our headers.
generate_export_header(tt_kurbla
    BASE_NAME TT_KURBLA
    EXPORT_MACRO_NAME TT_KURBLA_API
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/tt_kurbla_export.hpp
)

target_include_directories(tt_kurbla
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_compile_features(tt_kurbla PUBLIC cxx_std_20)

# Compile out log calls below the active level at build time.
# Debug/RelWithDebInfo: all levels compiled in (trace and above).
# Release and others: debug/trace calls become no-ops.
target_compile_definitions(tt_kurbla PUBLIC
    $<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE,
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
)


# tt-mlir's installed libraries are built without RTTI (LLVM convention), so
# any TU that instantiates tt-mlir's pipeline-options types (llvm::cl::Option,
# mlir::detail::PassOptions::OptionBase, ...) must also be compiled with
# -fno-rtti to avoid undefined typeinfo at link time. engine/compile.cpp
# instantiates TTIRToTTNNRuntimePipelineOptions, hence this flag.
#
# Applied PRIVATE: consumers (tests, future bindings) keep default RTTI today
# because nothing in our public headers forces them to construct such types.
# That's a property of the *current* public surface, not an enforced invariant
# — adding an inline header function or class that instantiates pipeline-
# options machinery would silently require consumers to also -fno-rtti.
# The existing test target catches that (it's built with default RTTI and
# uses the full public surface); if it ever fails to link with undefined
# typeinfo errors, the regression is in a public header.
target_compile_options(tt_kurbla PRIVATE -fno-rtti)

target_link_libraries(tt_kurbla
    PUBLIC
        tt_mlir::compiler
        tt_mlir::runtime
    PRIVATE
        tt-logger::tt-logger
        tt_kurbla_warnings
        tt_kurbla_sanitizers
)

# Tracy zone macros (`#include <tracy/Tracy.hpp>` + `ZoneScoped*`) are no-ops
# when TRACY_ENABLE is undefined, so the headers come in unconditionally.
# Linking libtracy.so and defining TRACY_ENABLE — the bits that actually emit
# zone code — are the only things gated by the option. Bundling of the Tracy
# CLI itself is independent of this option (see cmake/Tracy.cmake).
target_link_libraries(tt_kurbla PUBLIC tt_kurbla::tracy_headers)
if(TT_KURBLA_TRACY_ZONES)
    target_compile_definitions(tt_kurbla PUBLIC TRACY_ENABLE)
    target_link_libraries(tt_kurbla PUBLIC tt_kurbla::tracy)
endif()

if(TT_KURBLA_BUILD_PYTHON_EXT)
    add_subdirectory(torch)
endif()

OPTION(TT_KURBLA_BUILD_MAIN "If set, builds main.cpp for manual testing." OFF)
if (TT_KURBLA_BUILD_MAIN)
    add_executable(kurbla_main playground/main.cpp)

    target_link_libraries(kurbla_main
        PUBLIC
            tt_kurbla
            tt_mlir::compiler
            tt_mlir::runtime
        PRIVATE
            tt-logger::tt-logger
            tt_kurbla_warnings
            tt_kurbla_sanitizers
    )

    target_include_directories(kurbla_main
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
endif()
