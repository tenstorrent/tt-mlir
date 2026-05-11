configure_file(
    version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    @ONLY
)

add_library(tt_kurbla
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    engine/compile.cpp
)
add_library(tt_kurbla::tt_kurbla ALIAS tt_kurbla)

target_include_directories(tt_kurbla
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
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
