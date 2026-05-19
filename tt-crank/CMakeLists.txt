cmake_minimum_required(VERSION 3.24)

project(
    tt_kurbla
    VERSION 0.1.0
    DESCRIPTION "Compiler & runtime frontend for Tenstorrent hardware"
    LANGUAGES CXX
)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    enable_testing()
endif()

# Embed each target's link-path dirs into its INSTALL_RPATH automatically.
# Without this, the nanobind .so we install to site-packages can't find
# libtt_kurbla.so in the build tree at import time. Set before any add_subdirectory.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(TT_KURBLA_BUILD_TESTS "Build tt-kurbla unit tests" ${PROJECT_IS_TOP_LEVEL})
option(TT_KURBLA_BUILD_PYTHON_EXT
    "Build Python extension modules (set ON by scikit-build-core; OFF for C++-only builds)"
    OFF)

# Auto-source third_party/tt-mlir/env/activate so callers don't have to.
# No-op if already activated (e.g. by scripts/build). Must run before
# add_subdirectory(third_party), which checks $ENV{TTMLIR_TOOLCHAIN_DIR}.
include(cmake/MlirEnv.cmake)

include(cmake/Cache.cmake)
include(cmake/CompilerOptions.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/StaticAnalyzers.cmake)

add_subdirectory(third_party)
add_subdirectory(src)

# Must run after add_subdirectory(third_party) — the ttsim SoC-descriptor copy
# is wired as a build-time dependency on the tt-mlir-ep target.
include(cmake/TTsim.cmake)

if(TT_KURBLA_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
