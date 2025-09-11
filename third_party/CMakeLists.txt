include(ExternalProject)

set(TT_METAL_VERSION "e1308b0be2")

set(CMAKE_INSTALL_MESSAGE LAZY)  # suppress "Up-to-date:..." messages in incremental builds


if (DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
else()
  set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache")
endif()
message(STATUS "Setting tt-metal CPM cache to: ${CPM_SOURCE_CACHE}")

set(TRACY_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/tracy/public
)

set(TTMETAL_INCLUDE_DIRS
  ${TRACY_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/core
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/deprecated
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/include
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hostdevcommon/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/api
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/taskflow
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_stl
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_stl/tt_stl
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_eager
  ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build/include
  ${CPM_SOURCE_CACHE}/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0
  ${CPM_SOURCE_CACHE}/nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada/include
  ${CPM_SOURCE_CACHE}/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include
  ${CPM_SOURCE_CACHE}/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include
  ${CPM_SOURCE_CACHE}/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
  ${CPM_SOURCE_CACHE}/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
  ${CPM_SOURCE_CACHE}/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
  ${CPM_SOURCE_CACHE}/tt-logger/48ec77b63c35cf84c47678990aa3603b492d0c7c/include
  ${CPM_SOURCE_CACHE}/xtl/0_8_0_patched/include
  ${CPM_SOURCE_CACHE}/xtensor/0_26_0_patched/include
  ${CPM_SOURCE_CACHE}/xtensor-blas/0_22_0_patched/include
  PARENT_SCOPE
)

set(TTMETAL_BUILD_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build)
set(TTMETAL_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build/lib)

set(TTNN_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/_ttnncpp.so)
set(TTNN_PY_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/_ttnn.so)

set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtt_metal.so)
set(DEVICE_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libdevice.so)
set(TT_STL_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtt_stl.so)
if (TT_RUNTIME_ENABLE_PERF_TRACE)
  set(TRACY_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtracy.so)
else()
  set(TRACY_LIBRARY_PATH "")
endif()

set(TTMETAL_BUILD_DIR ${TTMETAL_BUILD_DIR} PARENT_SCOPE)
set(TTMETAL_LIBRARY_DIR ${TTMETAL_LIBRARY_DIR} PARENT_SCOPE)
set(TTNN_LIBRARY_PATH ${TTNN_LIBRARY_PATH} PARENT_SCOPE)
set(TTNN_PY_LIBRARY_PATH ${TTNN_PY_LIBRARY_PATH} PARENT_SCOPE)
set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_PATH} PARENT_SCOPE)
set(DEVICE_LIBRARY_PATH ${DEVICE_LIBRARY_PATH} PARENT_SCOPE)
set(TT_STL_LIBRARY_PATH ${TT_STL_LIBRARY_PATH} PARENT_SCOPE)
set(TRACY_LIBRARY_PATH ${TRACY_LIBRARY_PATH} PARENT_SCOPE)

if ("${CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "ccache")
  # `tt-metal` has specific setup for ccache.
  message(STATUS "Using ccache for tt-metal")
  set(TTMETAL_ENABLE_CCACHE ON)
  set(TTMETAL_DISABLE_PRECOMPILE_HEADERS ON)
endif()

if (TTMLIR_ENABLE_RUNTIME OR TTMLIR_ENABLE_OPMODEL)
set(TTMETAL_BUILD_TYPE "Release")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TTMETAL_BUILD_TYPE "Debug")
elseif (CMAKE_BUILD_TYPE STREQUAL "Asan")
  set(TTMETAL_BUILD_TYPE "ASan")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  set(ENABLE_DISTRIBUTED ON)
else()
  message(STATUS "Disabling tt-metal distributed support because of mpi x86 dep")
  set(ENABLE_DISTRIBUTED OFF)
endif()

set(METAL_BUILD_DIR ${TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build)
set(METAL_INSTALL_PREFIX "${METAL_BUILD_DIR}")
ExternalProject_Add(
  tt-metal
  PREFIX ${TTMLIR_SOURCE_DIR}/third_party/tt-metal
  BINARY_DIR ${METAL_BUILD_DIR}
  # This patch command creates the build dir that otherwise doesn't get created
  PATCH_COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_BUILD_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_MESSAGE=LAZY
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${TTMETAL_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${METAL_INSTALL_PREFIX}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}
    -DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}
    -DENABLE_CCACHE=${TTMETAL_ENABLE_CCACHE}
    -DWITH_PYTHON_BINDINGS=ON
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=${TTMETAL_DISABLE_PRECOMPILE_HEADERS}
    -DENABLE_TRACY=${TT_RUNTIME_ENABLE_PERF_TRACE}
    -DENABLE_LIBCXX=OFF
    -DENABLE_DISTRIBUTED=${ENABLE_DISTRIBUTED}
  GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
  GIT_TAG ${TT_METAL_VERSION}
  GIT_PROGRESS OFF
  BUILD_BYPRODUCTS ${TTNN_LIBRARY_PATH} ${TTNN_PY_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${TRACY_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH} ${TT_STL_LIBRARY_PATH}
)

ExternalProject_Add_StepTargets(tt-metal download update configure)
set_target_properties(tt-metal-download PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(tt-metal-update PROPERTIES EXCLUDE_FROM_ALL TRUE)
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
# We only need `tt_llk" and `umd` files from `third_party` directory.
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
  REGEX "third_party/tt_llk/tt_llk_blackhole/"
  REGEX "third_party/tt_llk/tt_llk_wormhole_b0/"
  REGEX "third_party/umd/"
)

# TTNN Standalone install commands
#
# Copy .cpmcache libs
install(
  DIRECTORY
    ${CPM_SOURCE_CACHE}/boost
    ${CPM_SOURCE_CACHE}/fmt
    ${CPM_SOURCE_CACHE}/enchantum
    ${CPM_SOURCE_CACHE}/nlohmann_json
    ${CPM_SOURCE_CACHE}/reflect
    ${CPM_SOURCE_CACHE}/spdlog
    ${CPM_SOURCE_CACHE}/tt-logger
  DESTINATION tt-metal/.cpmcache
  COMPONENT TTNNStandalone
  EXCLUDE_FROM_ALL
)
# Copy third_party
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party
  DESTINATION tt-metal/tt_metal
  COMPONENT TTNNStandalone
  EXCLUDE_FROM_ALL
)
# Copy tt_stl
install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_stl
  DESTINATION tt-metal
  COMPONENT TTNNStandalone
  EXCLUDE_FROM_ALL
)

list(APPEND library_names TTNN_LIBRARY TTNN_PY_LIBRARY TTMETAL_LIBRARY DEVICE_LIBRARY TT_STL_LIBRARY)
list(APPEND library_paths ${TTNN_LIBRARY_PATH} ${TTNN_PY_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH} ${TT_STL_LIBRARY_PATH})

if (TT_RUNTIME_ENABLE_PERF_TRACE)
  list(APPEND library_names TRACY_LIBRARY)
  list(APPEND library_paths ${TRACY_LIBRARY_PATH})
endif()

foreach(lib_name lib_path IN ZIP_LISTS library_names library_paths)
  add_library(${lib_name} SHARED IMPORTED GLOBAL)
  set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${lib_path})
  add_dependencies(${lib_name} tt-metal)
endforeach()

install(CODE
    "
    MESSAGE(STATUS \"Installing tt-metal shared lib dependencies (${library_paths}) to ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
    file(
        INSTALL
        ${library_paths}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
    )"
    COMPONENT SharedLib
)
else() # TTMLIR_ENABLE_RUNTIME OR TTMLIR_ENABLE_OPMODEL
  add_library(tt-metal INTERFACE)
  add_library(TTNN_LIBRARY INTERFACE)
  add_library(TTNN_PY_LIBRARY INTERFACE)
  add_library(TTMETAL_LIBRARY INTERFACE)
  add_library(DEVICE_LIBRARY INTERFACE)
  add_library(TT_STL_LIBRARY INTERFACE)
  add_library(tt-metal-download INTERFACE EXCLUDE_FROM_ALL)
  add_library(tt-metal-update INTERFACE EXCLUDE_FROM_ALL)
  add_library(tt-metal-configure INTERFACE EXCLUDE_FROM_ALL)
endif() # TTMLIR_ENABLE_RUNTIME OR TTMLIR_ENABLE_OPMODEL
