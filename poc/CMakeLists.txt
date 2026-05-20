cmake_minimum_required(VERSION 3.22)
project(tt-mlir-x280-poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Locate tt-metal + UMD. By default we use the in-tree tt-metal build that
# tt-mlir pulls in as a third_party dependency; override TT_METAL_BUILD on
# the command line to point at a different one.
# ---------------------------------------------------------------------------
get_filename_component(TT_MLIR_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
set(TT_METAL_BUILD "${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal/build"
    CACHE PATH "tt-metal build directory (must contain lib/cmake/umd/umdConfig.cmake)")

if(NOT EXISTS "${TT_METAL_BUILD}/lib/cmake/umd/umdConfig.cmake")
    message(FATAL_ERROR
        "Can't find umdConfig.cmake under ${TT_METAL_BUILD}/lib/cmake/umd. "
        "Pass -DTT_METAL_BUILD=/path/to/tt-metal/build or build tt-metal first.")
endif()

# tt-metal's plain build dir is awkward to consume directly: it ships stub
# *-config.cmake files at the build root that shadow the real configs in
# lib/cmake/<pkg>, and even when you skip those, find_dependency calls for
# transitive deps like nlohmann_json don't resolve. Prefer the standalone
# install tree under tt-mlir/build/tools/tt-alchemist/.../ttnn-install,
# which bundles every transitive dep with proper installed-layout configs.
get_filename_component(TT_MLIR_BUILD "${TT_MLIR_ROOT}/build" ABSOLUTE)
set(TTNN_INSTALL_DIR
    "${TT_MLIR_BUILD}/tools/tt-alchemist/templates/cpp/standalone/ttnn-install"
    CACHE PATH "Standalone TTNN install (overrides TT_METAL_BUILD for TTNN).")

list(PREPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD}/lib/cmake")
find_package(umd CONFIG REQUIRED)
# tt-logger is a transitive dep that umdConfig does not propagate explicitly.
find_package(tt-logger CONFIG REQUIRED)

# Optional: TTNN + Metalium for step3a_ttnn / step3 (TTNN integration).
# Switch to the bundled install for these find_packages.
if (EXISTS "${TTNN_INSTALL_DIR}/lib/cmake/tt-nn/tt-nn-config.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${TTNN_INSTALL_DIR}")
    find_package(tt-metalium CONFIG QUIET)
    find_package(TT-NN CONFIG QUIET)
endif()
if (TT-NN_FOUND AND tt-metalium_FOUND)
    set(POC_HAVE_TTNN TRUE)
else()
    set(POC_HAVE_TTNN FALSE)
    message(STATUS "TT-NN/Metalium not found at ${TTNN_INSTALL_DIR} — "
                   "skipping step3a_ttnn target.")
endif()

# ---------------------------------------------------------------------------
# Host helpers.
# ---------------------------------------------------------------------------
add_library(poc-common STATIC host/common.cpp)
target_include_directories(poc-common PUBLIC host)
target_link_libraries(poc-common PUBLIC umd::tt-umd tt-logger::tt-logger)

# ---------------------------------------------------------------------------
# Host binaries.
# ---------------------------------------------------------------------------
foreach(step step1_boot step2_task)
    add_executable(${step} host/${step}.cpp)
    target_link_libraries(${step} PRIVATE poc-common)
    set_target_properties(${step} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()

if (POC_HAVE_TTNN)
    foreach(step step3_ttnn_x280 step4_inplace step5_hoisted step6_multi_kernel)
        add_executable(${step} host/${step}.cpp)
        target_link_libraries(${step} PRIVATE TTNN::TTNN TT::Metalium poc-common)
        # TT_METAL_HOME / TT_METAL_RUNTIME_ROOT must point at the tt-metal
        # source tree at runtime so SoC descriptor YAMLs resolve. Bake the
        # absolute path in so callers don't have to source env/activate.
        target_compile_definitions(${step} PRIVATE
            POC_TT_METAL_HOME="${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal")
        set_target_properties(${step} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Firmware: cross-build via the riscv64-linux-gnu toolchain. We don't try to
# drive it through CMake's built-in cross-compile machinery because the host
# build already uses the native compiler and the firmware is a tiny leaf
# build — just shell out to the Makefile.
# ---------------------------------------------------------------------------
set(FW_SRC_DIR "${CMAKE_SOURCE_DIR}/fw")
set(FW_OUT_DIR "${CMAKE_BINARY_DIR}/fw")

file(MAKE_DIRECTORY "${FW_OUT_DIR}")

add_custom_command(
    OUTPUT "${FW_OUT_DIR}/fw_step1.bin" "${FW_OUT_DIR}/fw_step2.bin"
           "${FW_OUT_DIR}/fw_step4.bin" "${FW_OUT_DIR}/fw_step5.bin"
           "${FW_OUT_DIR}/fw_step6.bin" "${FW_OUT_DIR}/cpu2_blob.bin"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${FW_SRC_DIR}" "${FW_OUT_DIR}"
    COMMAND make -C "${FW_OUT_DIR}"
    DEPENDS
        "${FW_SRC_DIR}/fw.ld"
        "${FW_SRC_DIR}/fw_start.S"
        "${FW_SRC_DIR}/fw_step1.c"
        "${FW_SRC_DIR}/fw_step2.c"
        "${FW_SRC_DIR}/fw_step4.c"
        "${FW_SRC_DIR}/fw_step5.c"
        "${FW_SRC_DIR}/fw_step6.c"
        "${FW_SRC_DIR}/cpu.o"
        "${FW_SRC_DIR}/cpu2.ll"
        "${FW_SRC_DIR}/cpu2_dispatch.c"
        "${FW_SRC_DIR}/cpu2_blob.ld"
        "${FW_SRC_DIR}/Makefile"
    COMMENT "Cross-compiling X280 firmware"
    VERBATIM)

add_custom_target(firmware ALL
    DEPENDS "${FW_OUT_DIR}/fw_step1.bin" "${FW_OUT_DIR}/fw_step2.bin"
            "${FW_OUT_DIR}/fw_step4.bin" "${FW_OUT_DIR}/fw_step5.bin"
            "${FW_OUT_DIR}/fw_step6.bin" "${FW_OUT_DIR}/cpu2_blob.bin")
