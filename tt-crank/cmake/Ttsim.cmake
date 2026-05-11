# Stages a tenstorrent/ttsim simulator next to its SoC descriptor in the layout
# tt-metal expects ($TT_METAL_SIMULATOR_HOME = a dir with libttsim.so +
# soc_descriptor.yaml). Mirrors what tt-mlir's call-test-ttsim.yml workflow
# does in shell.
#
# Exposes (parent scope) when TT_KURBLA_ENABLE_SIMULATOR=ON:
#   TT_KURBLA_SIM_DIR          staged sim home dir
#   TT_KURBLA_TT_METAL_HOME    tt-metal source tree the runtime needs at TT_METAL_RUNTIME_ROOT

option(TT_KURBLA_ENABLE_SIMULATOR
    "Download ttsim and route runtime tests through the simulator instead of opening a physical device"
    OFF)

set(TT_KURBLA_TTSIM_VERSION "v1.5.1" CACHE STRING
    "ttsim release tag (https://github.com/tenstorrent/ttsim/releases)")

if(NOT TT_KURBLA_ENABLE_SIMULATOR)
    return()
endif()

set(TT_KURBLA_SIM_ARCH "wh" CACHE STRING "Simulator arch (wh|bh)")
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
# data files (soc descriptors, firmware sources) that ttsim needs.
set(TT_KURBLA_TT_METAL_HOME "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal")
set(_soc_src "${TT_KURBLA_TT_METAL_HOME}/tt_metal/soc_descriptors/${_soc_filename}")
if(NOT EXISTS "${_soc_src}")
    message(FATAL_ERROR "Missing SoC descriptor ${_soc_src}. "
        "Run `git submodule update --init --recursive` and a full build first to populate tt-metal.")
endif()

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

# Re-run on cache change; copies under the names tt-metal expects.
configure_file("${_lib_cache}" "${TT_KURBLA_SIM_DIR}/libttsim.so"          COPYONLY)
configure_file("${_soc_src}"   "${TT_KURBLA_SIM_DIR}/soc_descriptor.yaml" COPYONLY)
