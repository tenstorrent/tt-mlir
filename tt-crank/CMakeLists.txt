cmake_minimum_required(VERSION 3.24)

project(
    tt_crank
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

option(TT_CRANK_WHEEL "Configure the build for a distributable Python wheel" OFF)

# Dev builds resolve deps from the build tree; wheels set $ORIGIN per target.
# Must precede add_subdirectory — targets capture this at creation.
if(NOT TT_CRANK_WHEEL)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# Silence the per-file "-- Installing:"/"-- Up-to-date:" messages on install.
# Set before any add_subdirectory so it applies to third-party install rules too.
set(CMAKE_INSTALL_MESSAGE NEVER)

option(TT_CRANK_BUILD_TESTS "Build tt-crank unit tests" ${PROJECT_IS_TOP_LEVEL})
option(TT_CRANK_BUILD_PYTHON_EXT
    "Build Python extension modules (set ON by scikit-build-core; OFF for C++-only builds)"
    OFF)
# Decoupled from CLI bundling: the Tracy CLI (`tracy`, tracy-capture,
# tracy-csvexport) is always built and shipped with tt-crank. This flag
# only controls whether libtt_crank itself is instrumented with ZoneScoped
# macros.
option(TT_CRANK_TRACY_ZONES
    "Compile Tracy zones into libtt_crank (TRACY_ENABLE + link libtracy)"
    ON)

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
include(cmake/Tracy.cmake)

if(TT_CRANK_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
