include(GenerateExportHeader)

add_subdirectory(common)

# Bake tt-mlir provenance into version.cpp: the plain commit hash
# (ttmlir_git_hash()) and a combined commit+worktree hash
# (ttmlir_git_worktree_hash(), == commit when clean) for keying a compile cache.
include(${PROJECT_SOURCE_DIR}/cmake/GitRev.cmake)

set(TT_CRANK_TTMLIR_SOURCE_DIR "${CMAKE_SOURCE_DIR}")

git_commit_hash(TTMLIR_GIT_HASH ${TT_CRANK_TTMLIR_SOURCE_DIR})
if(TTMLIR_GIT_HASH STREQUAL "")
    message(FATAL_ERROR
        "Could not resolve the tt-mlir commit hash from "
        "${TT_CRANK_TTMLIR_SOURCE_DIR}. Initialize the submodule "
        "(git submodule update --init --recursive), or point "
        "-DTTMLIR_SOURCE_DIR_OVERRIDE at a git checkout.")
endif()

git_worktree_hash(TTMLIR_GIT_WORKTREE_HASH ${TT_CRANK_TTMLIR_SOURCE_DIR})
if(TTMLIR_GIT_WORKTREE_HASH STREQUAL "")
    message(FATAL_ERROR
        "Could not resolve the tt-mlir worktree hash from "
        "${TT_CRANK_TTMLIR_SOURCE_DIR} ('git diff HEAD' failed).")
endif()

configure_file(
    version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    @ONLY
)

add_library(tt_crank SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    engine/compile.cpp
    engine/compile_options.cpp
    engine/device.cpp
    engine/execution_payload.cpp
    engine/ttir_module_builder.cpp
    # Static-init env shim runtime. Linked into the library so every consumer (tests,
    # Python wheel, future bindings) picks up the ttsim routing when the env var is set.
    # TT_CRANK_SIM_DIR / TT_CRANK_TT_METAL_HOME defs are wired in cmake/TTsim.cmake.
    engine/sim_env.cpp
)
add_library(tt_crank::tt_crank ALIAS tt_crank)

# Hide every symbol by default; only declarations marked TT_CRANK_API end up in
# the .so's dynamic symbol table. Without this, every namespace-scope symbol
# (including unnamed-namespace helpers and tt-mlir template instantiations
# reachable from our headers) would leak into the public ABI on Linux.
set_target_properties(tt_crank PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Generates tt_crank_export.hpp defining TT_CRANK_API (plus deprecation
# helpers). The macro expands to the export attribute when building tt_crank
# itself (tt_crank_EXPORTS is auto-defined by CMake for shared targets) and to
# the import attribute when consumers compile against our headers.
generate_export_header(tt_crank
    BASE_NAME TT_CRANK
    EXPORT_MACRO_NAME TT_CRANK_API
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/tt_crank_export.hpp
)

target_include_directories(tt_crank
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_compile_features(tt_crank PUBLIC cxx_std_20)

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
target_compile_options(tt_crank PRIVATE -fno-rtti)

target_link_libraries(tt_crank
    PUBLIC
        tt_mlir::compiler
        tt_mlir::runtime
        tt_crank_common
    PRIVATE
        tt_crank_warnings
        tt_crank_sanitizers
)

# We include `tracy` files unconditionally, so add proper include dirs.
target_include_directories(tt_crank SYSTEM PRIVATE ${TRACY_INCLUDE_DIRS})
if(TT_CRANK_TRACY_ZONES)
    # TODO(pilkic): These compile definitions are set globally in `tt-mlir`. There
    # should be a better way...
    # target_compile_definitions(tt_crank PUBLIC TRACY_ENABLE TRACY_TIMER_FALLBACK)
    target_link_libraries(tt_crank PUBLIC tt_metal::tracy)
endif()

# Wheel: libtt_crank.so lands in tt_crank/lib alongside its deps (tt-mlir's libs,
# libtt_crank_common.so), so $ORIGIN resolves everything. RPATH must be decided at
# configure time; what ships is decided by the component.
if(TT_CRANK_WHEEL)
    set_target_properties(tt_crank PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

install(TARGETS tt_crank
    LIBRARY
        DESTINATION tt_crank/lib
        COMPONENT wheel_payload
        EXCLUDE_FROM_ALL
)

if(TT_CRANK_BUILD_PYTHON_EXT)
    add_subdirectory(torch)
endif()

OPTION(TT_CRANK_BUILD_MAIN "If set, builds main.cpp for manual testing." OFF)
if (TT_CRANK_BUILD_MAIN)
    add_executable(crank_main playground/main.cpp)

    target_link_libraries(crank_main
        PUBLIC
            tt_crank
            tt_mlir::compiler
            tt_mlir::runtime
            tt_crank_common
        PRIVATE
            tt_crank_warnings
            tt_crank_sanitizers
    )

    target_include_directories(crank_main
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
endif()
