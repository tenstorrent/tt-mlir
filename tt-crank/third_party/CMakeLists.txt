include(${CMAKE_SOURCE_DIR}/cmake/SystemIncludes.cmake)

include(FetchContent)
FetchContent_Declare(
    tt-logger
    GIT_REPOSITORY https://github.com/tenstorrent/tt-logger.git
    GIT_TAG        v1.1.8
)
set(SPDLOG_BUILD_PIC ON CACHE INTERNAL "")
FetchContent_MakeAvailable(tt-logger)

mark_as_system_includes(tt-logger spdlog_header_only fmt-header-only)

# Build tt-mlir as a separate, isolated CMake invocation (ExternalProject_Add)
# rather than add_subdirectory'ing it. Two reasons:
#   - tt-mlir's CMakeLists.txt forces CMAKE_C/CXX_COMPILER=clang and requires
#     TTMLIR_ENV_ACTIVATED at configure time; embedding it would constrain our
#     parent build.
#   - tt-mlir's configure is slow; isolating it keeps incremental builds of
#     tt-crank fast.
#
# Downstream consumes the installed shared libs + headers via IMPORTED targets
# `tt_mlir::compiler` (libTTMLIRCompiler.so) and `tt_mlir::runtime`
# (libTTMLIRRuntime.so). Both share the same install include tree.

include(ExternalProject)

set(TTMLIR_SOURCE_DIR_OVERRIDE "" CACHE PATH
    "Path to an external tt-mlir source tree (overrides the submodule). Useful when iterating on tt-mlir in a sibling checkout.")

if(TTMLIR_SOURCE_DIR_OVERRIDE)
    set(_TTMLIR_SRC "${TTMLIR_SOURCE_DIR_OVERRIDE}")
else()
    set(_TTMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
endif()

if(NOT EXISTS "${_TTMLIR_SRC}/CMakeLists.txt")
    message(FATAL_ERROR
        "tt-mlir source not found at ${_TTMLIR_SRC}. "
        "Run `git submodule update --init --recursive` "
        "or pass `-DTTMLIR_SOURCE_DIR_OVERRIDE=<path>`.")
endif()

if(NOT DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
    message(FATAL_ERROR
        "TTMLIR_TOOLCHAIN_DIR is not set in the environment. "
        "Run `source third_party/tt-mlir/env/activate` from tt-mlir's root "
        "(or use ./scripts/build) before configuring tt-crank.")
endif()

# Promoted to PARENT_SCOPE at the end of this file so cmake/Tracy.cmake (and
# other top-level helpers) can reach into the install dir.
set(_TTMLIR_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/tt-mlir-install")
set(_TTMLIR_COMPILER_LIB "${_TTMLIR_INSTALL_DIR}/lib/libTTMLIRCompiler.so")
set(_TTMLIR_RUNTIME_LIB  "${_TTMLIR_INSTALL_DIR}/lib/libTTMLIRRuntime.so")
set(_TTMLIR_TRACY_LIB    "${_TTMLIR_INSTALL_DIR}/lib/libtracy.so")

# CMake requires INTERFACE_INCLUDE_DIRECTORIES paths on IMPORTED targets to exist
# at configure time. The directory will be populated by tt-mlir's install step.
file(MAKE_DIRECTORY "${_TTMLIR_INSTALL_DIR}/include")

# CMAKE_ARGS lifted out of ExternalProject_Add because we override
# CONFIGURE_COMMAND below (which would otherwise silently drop them).
set(_TTMLIR_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${_TTMLIR_INSTALL_DIR}
    # Silence the per-file "-- Installing:"/"-- Up-to-date:" messages.
    -DCMAKE_INSTALL_MESSAGE=NEVER
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DTTMLIR_ENABLE_RUNTIME=ON
    -DTTMLIR_ENABLE_SHARED_LIB=ON
    -DTTMLIR_ENABLE_STABLEHLO=OFF
    -DTTMLIR_ENABLE_TESTS=OFF
    -DTTMLIR_ENABLE_TOOLS=OFF
    -DTTMLIR_ENABLE_TTRT=OFF
    -DTTMLIR_ENABLE_OPMODEL=ON
    -DTTMLIR_ENABLE_OPMODEL_TESTS=OFF
    -DTTMLIR_ENABLE_EXPLORER=OFF
    -DTTMLIR_ENABLE_ALCHEMIST=OFF
    -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF
    # Always build tt-metal/tt-mlir with Tracy. The Tracy CLI + tracy-capture
    # binaries are bundled into every tt-crank build via cmake/Tracy.cmake;
    # whether libtt_crank itself is instrumented is a separate switch
    # (TT_CRANK_TRACY_ZONES, defined in the top-level CMakeLists.txt).
    -DTT_RUNTIME_ENABLE_PERF_TRACE=ON
)

# Inject env/activate's variables (collected by cmake/MlirEnv.cmake) into
# every tt-mlir-ep subprocess. set(ENV{}) at our configure time only affects
# the parent cmake process; tt-mlir's own cmake invocation runs as a
# subprocess at build time, inheriting env from ninja — which doesn't see
# anything we set. `cmake -E env KEY=VAL ... cmd args...` is the per-call fix.
ExternalProject_Add(tt-mlir-ep
    SOURCE_DIR  "${_TTMLIR_SRC}"
    PREFIX      "${CMAKE_CURRENT_BINARY_DIR}/tt-mlir-prefix"
    INSTALL_DIR "${_TTMLIR_INSTALL_DIR}"
    CMAKE_GENERATOR Ninja
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${TT_CRANK_MLIR_ENV}
        ${CMAKE_COMMAND} -GNinja ${_TTMLIR_CMAKE_ARGS}
        -S <SOURCE_DIR> -B <BINARY_DIR>
    BUILD_COMMAND ${CMAKE_COMMAND} -E env ${TT_CRANK_MLIR_ENV}
        ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} -E env ${TT_CRANK_MLIR_ENV}
        ${CMAKE_COMMAND} --install <BINARY_DIR> --component SharedLib
    COMMAND ${CMAKE_COMMAND} -E env ${TT_CRANK_MLIR_ENV}
        ${CMAKE_COMMAND} --install <BINARY_DIR> --component DistributedRuntime
    # Declare the install outputs we consume so Ninja can wire the dependency
    # graph; without this it errors with "no known rule to make" on link.
    BUILD_BYPRODUCTS
        "${_TTMLIR_COMPILER_LIB}"
        "${_TTMLIR_RUNTIME_LIB}"
        "${_TTMLIR_TRACY_LIB}"
    BUILD_ALWAYS FALSE
    USES_TERMINAL_BUILD TRUE
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
)

# tt-mlir's installed headers transitively include `<mlir-c/...>` headers, which
# live in the LLVM/MLIR toolchain prefix — not in tt-mlir's own install dir.
# Surface both on the IMPORTED target so consumers see them with one link.
set(_TTMLIR_INCLUDE_DIRS
    "${_TTMLIR_INSTALL_DIR}/include"
    "$ENV{TTMLIR_TOOLCHAIN_DIR}/include"
)

# IMPORTED targets wrapping the installed shared libraries.
add_library(tt_mlir::compiler SHARED IMPORTED GLOBAL)
add_dependencies(tt_mlir::compiler tt-mlir-ep)
set_target_properties(tt_mlir::compiler PROPERTIES
    IMPORTED_LOCATION "${_TTMLIR_COMPILER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_TTMLIR_INCLUDE_DIRS}"
)

add_library(tt_mlir::runtime SHARED IMPORTED GLOBAL)
add_dependencies(tt_mlir::runtime tt-mlir-ep)
set_target_properties(tt_mlir::runtime PROPERTIES
    IMPORTED_LOCATION "${_TTMLIR_RUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_TTMLIR_INCLUDE_DIRS}"
)

# Treat tt-mlir + MLIR headers as SYSTEM so our warnings/clang-tidy don't fire on them.
mark_as_system_includes(tt_mlir::compiler tt_mlir::runtime)

# libtracy.so is built by tt-metal (`OUTPUT_NAME "tracy"`, see
# tt-metal/cmake/tracy.cmake) and installed by tt-mlir into lib/ under the
# SharedLib component.
#
# The Tracy public headers are *not* in the install tree - we pick
# them up directly from the tt-metal clone that tt-mlir-ep populates.
set(_TRACY_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/tracy/public")
file(MAKE_DIRECTORY "${_TRACY_INCLUDE_DIR}")

add_library(tt_crank_tracy_headers INTERFACE)
target_include_directories(tt_crank_tracy_headers SYSTEM INTERFACE "${_TRACY_INCLUDE_DIR}")
add_dependencies(tt_crank_tracy_headers tt-mlir-ep)
add_library(tt_crank::tracy_headers ALIAS tt_crank_tracy_headers)

add_library(tt_crank::tracy SHARED IMPORTED GLOBAL)
add_dependencies(tt_crank::tracy tt-mlir-ep)
set_target_properties(tt_crank::tracy PROPERTIES
    IMPORTED_LOCATION "${_TTMLIR_TRACY_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_TRACY_INCLUDE_DIR}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_TRACY_INCLUDE_DIR}"
)

# Expose the source + install dirs to the parent CMake scope.
set(TT_CRANK_TTMLIR_SOURCE_DIR  "${_TTMLIR_SRC}" PARENT_SCOPE)
set(TT_CRANK_TTMLIR_INSTALL_DIR "${_TTMLIR_INSTALL_DIR}" PARENT_SCOPE)

# ---------------------------------------------------------------------------
# Wheel payload: harvest the tt-mlir install tree. scikit-build-core turns these
# install() rules into wheel platlib contents. All tagged `COMPONENT
# wheel_payload EXCLUDE_FROM_ALL` because editable installs also run
# `cmake --install` — keep those keywords BEFORE any FILES_MATCHING/PATTERN,
# which swallow every argument after them.
#
# NOTE the wheel writer dereferences symlinks, so soname chains like
# libtt-umd.so.0 -> .0.74.0 materialize as concrete files.

# (a) Runtime shared libraries. .so only: skips ~140 MB of .a archives and the
#     lib/cmake export files.
install(DIRECTORY "${_TTMLIR_INSTALL_DIR}/lib/"
    DESTINATION "tt_crank/lib"
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    FILES_MATCHING PATTERN "*.so*")

# (b) All of bin/: the Tracy CLI pair that python/tracy points PROFILER_BIN_DIR
#     at, plus ttmlir/runtime/distributed/worker.
install(DIRECTORY "${_TTMLIR_INSTALL_DIR}/bin/"
    DESTINATION "tt_crank/bin"
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL)

# (c) tt-metal tree, resolved as TT_METAL_HOME at runtime. Excludes sfpi
#     (provisioned separately by `tt-crank-install-sfpi`) and _ttnn.so
#     (shipped from tt_crank/lib).
install(DIRECTORY "${_TTMLIR_INSTALL_DIR}/tt-metal/"
    DESTINATION "tt_crank/tt-metal"
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    PATTERN "sfpi" EXCLUDE
    PATTERN "_ttnn.so" EXCLUDE)

# (d) Re-export _original trees as REAL directories — the wheel writer can't carry
#     the build-tree symlinks from cmake/Tracy.cmake. python/{ttnn,tracy} import
#     from these; _wrapper.py loads _ttnn.so from tt_crank/lib instead.
install(DIRECTORY "${_TTMLIR_INSTALL_DIR}/tt-metal/ttnn/ttnn/"
    DESTINATION "ttnn/_original"
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    PATTERN "_ttnn.so" EXCLUDE)
install(DIRECTORY "${_TTMLIR_INSTALL_DIR}/tt-metal/tools/tracy/"
    DESTINATION "tracy/_original"
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL)
