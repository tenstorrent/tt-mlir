# Stages a tenstorrent/ttsim simulator next to its SoC descriptor in the layout
# tt-metal expects ($TT_METAL_SIMULATOR_HOME = a dir with libttsim.so +
# soc_descriptor.yaml). Mirrors what tt-mlir's call-test-ttsim.yml workflow
# does in shell.
#
# Always builds the staging and wires the paths into libtt_kurbla.so. Whether
# tt-kurbla actually opens ttsim is a runtime decision driven by the
# TT_KURBLA_USE_SIMULATOR env var — see src/engine/sim_env.cpp and the `sim`
# test preset in CMakePresets.json.
#
# Exposes (parent scope):
#   TT_KURBLA_SIM_DIR          staged sim home dir
#   TT_KURBLA_TT_METAL_HOME    tt-metal source tree the runtime needs at TT_METAL_RUNTIME_ROOT
#   tt_kurbla_ttsim_stage      custom target — depend on this from anything that
#                              needs the staged dir to be populated at build time

set(TT_KURBLA_TTSIM_VERSION "v1.10.1" CACHE STRING
    "ttsim release tag (https://github.com/tenstorrent/ttsim/releases)")

# Default to blackhole: it provides more coverage, since wormhole kernels have more
# issues which get exposed when running through the sim.
set(TT_KURBLA_SIM_ARCH "bh" CACHE STRING "Simulator arch (wh|bh)")
set_property(CACHE TT_KURBLA_SIM_ARCH PROPERTY STRINGS wh bh)

if(TT_KURBLA_SIM_ARCH STREQUAL "wh")
    set(_soc_filename "wormhole_b0_80_arch.yaml")
elseif(TT_KURBLA_SIM_ARCH STREQUAL "bh")
    set(_soc_filename "blackhole_140_arch.yaml")
else()
    message(FATAL_ERROR "TT_KURBLA_SIM_ARCH must be 'wh' or 'bh', got '${TT_KURBLA_SIM_ARCH}'")
endif()

# tt-metal source tree, vendored under tt-mlir's submodule. Source — not the
# install dir — because tt-mlir's ExternalProject doesn't install the runtime
# data files (soc descriptors, firmware sources) that ttsim needs. Populated
# by tt-mlir-ep's build step, so the descriptor copy is deferred to build time.
set(TT_KURBLA_TT_METAL_HOME "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal")
set(_soc_src "${TT_KURBLA_TT_METAL_HOME}/tt_metal/soc_descriptors/${_soc_filename}")

set(TT_KURBLA_SIM_DIR "${CMAKE_BINARY_DIR}/ttsim_home")
file(MAKE_DIRECTORY "${TT_KURBLA_SIM_DIR}")

# Cache the download by arch so flipping TT_KURBLA_SIM_ARCH doesn't redownload.
set(_lib_cache "${CMAKE_BINARY_DIR}/_deps/libttsim_${TT_KURBLA_SIM_ARCH}_${TT_KURBLA_TTSIM_VERSION}.so")
if(NOT EXISTS "${_lib_cache}")
    message(STATUS "Downloading ttsim ${TT_KURBLA_TTSIM_VERSION} (${TT_KURBLA_SIM_ARCH})")
    file(DOWNLOAD
        "https://github.com/tenstorrent/ttsim/releases/download/${TT_KURBLA_TTSIM_VERSION}/libttsim_${TT_KURBLA_SIM_ARCH}.so"
        "${_lib_cache}"
        STATUS _status SHOW_PROGRESS TLS_VERIFY ON)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
        file(REMOVE "${_lib_cache}")
        list(GET _status 1 _msg)
        message(FATAL_ERROR "ttsim download failed: ${_msg}")
    endif()
endif()

# libttsim.so has no build-tree dependency — stage at configure time.
configure_file("${_lib_cache}" "${TT_KURBLA_SIM_DIR}/libttsim.so" COPYONLY)

# The SoC descriptor lives in tt-metal's source tree (downloaded by tt-mlir-ep
# at build time), so we can't copy it at configure time on a fresh clone.
# Defer to a build-time custom command that depends on tt-mlir-ep having run.
set(_soc_staged "${TT_KURBLA_SIM_DIR}/soc_descriptor.yaml")
add_custom_command(
    OUTPUT "${_soc_staged}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_soc_src}" "${_soc_staged}"
    DEPENDS tt-mlir-ep
    COMMENT "Staging ttsim SoC descriptor (${TT_KURBLA_SIM_ARCH})"
    VERBATIM
)
add_custom_target(tt_kurbla_ttsim_stage DEPENDS "${_soc_staged}")

# Wire the sim shim into the library: paths via compile defs, staged-dir
# population via a build-time dependency. Done here (not in src/CMakeLists.txt)
# because TTsim.cmake is the source of truth for these paths and it's included
# after add_subdirectory(src).
target_compile_definitions(tt_kurbla PRIVATE
    TT_KURBLA_SIM_DIR="${TT_KURBLA_SIM_DIR}"
    TT_KURBLA_TT_METAL_HOME="${TT_KURBLA_TT_METAL_HOME}"
)
add_dependencies(tt_kurbla tt_kurbla_ttsim_stage)
