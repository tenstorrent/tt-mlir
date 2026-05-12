include(GenerateExportHeader)

configure_file(
    version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    @ONLY
)

add_library(tt_kurbla SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    engine/compile.cpp
    engine/execution_payload.cpp
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
        tt_kurbla_warnings
        tt_kurbla_sanitizers
)

if(TT_KURBLA_BUILD_PYTHON_EXT)
    add_subdirectory(torch)
endif()
