include(ExternalProject)

set(TT_METAL_VERSION "9d69fb143bac50983dff914c5348539d0a7d2021")

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

set(METAL_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/")
set(TTMETAL_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/deprecated
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hostdevcommon/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_NAME}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/tracy/public
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/taskflow
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_eager
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/boost_core/e679bef5c160cf29d0f37d549881dc5f5a58c332/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/json/798e0374658476027d9723eeb67a262d0f3c8308/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtensor/4a957e26c765b48cbec4a4235fe9e518d5a85d3d/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtensor-blas/190c3a4314355b67291a7d78b20a2100de3f8f54/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtl/0918808959d33a292c551b9f014a0e808bc4a95c/include
  PARENT_SCOPE
)

set(TTMETAL_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/lib)
set(TTNN_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/_ttnn.so)
set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtt_metal.so)
set(DEVICE_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libdevice.so)
if (TT_RUNTIME_ENABLE_PERF_TRACE)
  set(TRACY_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtracy.so)
  # set(TRACY_LIBRARY_VERSIONED_PATH ${TTMETAL_LIBRARY_DIR}/libtracy.so.0.10.0)
else()
  set(TRACY_LIBRARY_PATH "")
  # set(TRACY_LIBRARY_VERSIONED_PATH "")
endif()

set(TTMETAL_LIBRARY_DIR ${TTMETAL_LIBRARY_DIR} PARENT_SCOPE)
set(TTNN_LIBRARY_PATH ${TTNN_LIBRARY_PATH} PARENT_SCOPE)
set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_PATH} PARENT_SCOPE)
set(DEVICE_LIBRARY_PATH ${DEVICE_LIBRARY_PATH} PARENT_SCOPE)
set(TRACY_LIBRARY_PATH ${TRACY_LIBRARY_PATH} PARENT_SCOPE)
# set(TRACY_LIBRARY_VERSIONED_PATH ${TRACY_LIBRARY_VERSIONED_PATH} PARENT_SCOPE)

ExternalProject_Add(
  tt-metal
  PREFIX ${TTMLIR_SOURCE_DIR}/third_party/tt-metal
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${METAL_INSTALL_PREFIX}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DENABLE_TRACY=${TT_RUNTIME_ENABLE_PERF_TRACE}
    -DENABLE_LIBCXX=OFF
  GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
  GIT_TAG ${TT_METAL_VERSION}
  GIT_PROGRESS ON
  BUILD_BYPRODUCTS ${TTNN_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${TRACY_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH}
)

ExternalProject_Add_StepTargets(tt-metal download configure)
set_target_properties(tt-metal-download PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(tt-metal-configure PROPERTIES EXCLUDE_FROM_ALL TRUE)

set_target_properties(tt-metal PROPERTIES EXCLUDE_FROM_ALL TRUE)

# When installing, copy all needed files (for runtime) from `tt-metal` repo to the install directory.
# EXCLUDE_FROM_ALL is used to avoid installing these files when installing the default component.
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn
  DESTINATION ${CMAKE_INSTALL_PREFIX}/tt-metal
  COMPONENT SharedLib
  EXCLUDE_FROM_ALL
  REGEX "ttnn/ttnn" EXCLUDE
)
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/runtime
  DESTINATION ${CMAKE_INSTALL_PREFIX}/tt-metal
  USE_SOURCE_PERMISSIONS
  COMPONENT SharedLib
  EXCLUDE_FROM_ALL
)

# `tt_metal` directory is installed with two commands, since we need special handling of `third_party` directory.
# We only need `tt_llk_*" and `umd` files from `third_party` directory.
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
  DESTINATION ${CMAKE_INSTALL_PREFIX}/tt-metal
  COMPONENT SharedLib
  EXCLUDE_FROM_ALL
  PATTERN "third_party" EXCLUDE
)

# NOTE: For some reason, CMake creates all directories even for non-matched files, but those will be empty.
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
  DESTINATION ${CMAKE_INSTALL_PREFIX}/tt-metal
  COMPONENT SharedLib
  EXCLUDE_FROM_ALL
  FILES_MATCHING
  REGEX "third_party/tt_llk_*"
  REGEX "third_party/umd/"
)

list(APPEND library_names TTNN_LIBRARY TTMETAL_LIBRARY DEVICE_LIBRARY)
list(APPEND library_paths ${TTNN_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH})



foreach(lib_name lib_path IN ZIP_LISTS library_names library_paths)
  add_library(${lib_name} SHARED IMPORTED GLOBAL)
  set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${lib_path})
  add_dependencies(${lib_name} tt-metal)
endforeach()

install(FILES ${library_paths} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)
if (TT_RUNTIME_ENABLE_PERF_TRACE)
  add_library(TRACY_LIBRARY TRACY_LIBRARY_PATH SHARED IMPORTED GLOBAL)
  add_dependencies(TRACY_LIBRARY tt-metal)
  install(CODE
      "
      MESSAGE(STATUS \"Installing tt-mlir runtime shared lib dependencies (${TRACY_LIBRARY_PATH}) to ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
      file(
          INSTALL
          ${TRACY_LIBRARY_PATH}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          TYPE SHARED_LIBRARY
          FOLLOW_SYMLINK_CHAIN
      )"
      COMPONENT SharedLib
      EXCLUDE_FROM_ALL
  )
endif()
