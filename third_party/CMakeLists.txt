include(ExternalProject)

if ("$ENV{ARCH_NAME}" STREQUAL "grayskull")
set(ARCH_NAME "grayskull")
set(ARCH_EXTRA_DIR "grayskull")
elseif ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
set(ARCH_NAME "wormhole")
set(ARCH_EXTRA_DIR "wormhole/wormhole_b0_defines")
elseif ("$ENV{ARCH_NAME}" STREQUAL "blackhole")
set(ARCH_NAME "blackhole")
set(ARCH_EXTRA_DIR "blackhole")
else()
message(FATAL_ERROR "Unsupported ARCH_NAME: $ENV{ARCH_NAME}")
endif()

set(TTMETAL_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/fmt
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_NAME}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_eager
  PARENT_SCOPE
)
set(TTMETAL_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/lib PARENT_SCOPE)
set(TTMETAL_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/lib)

set(TTNN_LIBRARY ${TTMETAL_LIBRARY_DIR}/_ttnn.so PARENT_SCOPE)
set(TTNN_LIBRARY ${TTMETAL_LIBRARY_DIR}/_ttnn.so)

set(TTMETAL_LIBRARY -ltt_metal PARENT_SCOPE)

if(EXISTS ${TTNN_LIBRARY})
message(STATUS "Found TTNN library at ${TTNN_LIBRARY}")
add_custom_target(tt-metal)
else()
ExternalProject_Add(
    tt-metal
    PREFIX ${TTMLIR_SOURCE_DIR}/third_party/tt-metal
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
    GIT_TAG v0.49.0
    GIT_PROGRESS ON
)
endif()
set_target_properties(tt-metal PROPERTIES EXCLUDE_FROM_ALL TRUE)
