cmake_minimum_required(VERSION 3.20.0)
project(ttmlir-toolchain LANGUAGES CXX C)
include(ExternalProject)

set(TTMLIR_TOOLCHAIN "/opt/ttmlir-toolchain" CACHE STRING "Path to the toolchain")

if (NOT EXISTS "${TTMLIR_TOOLCHAIN}")
  message( FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTMLIR_TOOLCHAIN}\n  sudo chown -R $ENV{USER}:wheel ${TTMLIR_TOOLCHAIN}")
endif()

add_custom_target(python-venv ALL COMMAND TTMLIR_TOOLCHAIN=${TTMLIR_TOOLCHAIN} CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} bash ${CMAKE_CURRENT_SOURCE_DIR}/build_venv.sh)

ExternalProject_Add(
    flatbuffers
    PREFIX ${TTMLIR_TOOLCHAIN}
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN}
      -DFLATBUFFERS_BUILD_TESTS=OFF
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.7
    DEPENDS python-venv
)

# -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON set if debug
ExternalProject_Add(
    llvm-project
    PREFIX ${TTMLIR_TOOLCHAIN}
    # Super hacky way to install the python dependencies before the build
    PATCH_COMMAND bash -c "source ${CMAKE_CURRENT_SOURCE_DIR}/activate && pip install -r ${TTMLIR_TOOLCHAIN}/src/llvm-project/mlir/python/requirements.txt"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DPython3_FIND_VIRTUALENV=ONLY
      -DPython3_EXECUTABLE=${TTMLIR_TOOLCHAIN}/venv/bin/python
      -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN}
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_INSTALL_UTILS=ON
      -DCMAKE_BUILD_TYPE=MinSizeRel
      -DLLVM_ENABLE_ASSERTIONS=ON
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON
      -DCMAKE_C_FLAGS=-D_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION
    SOURCE_SUBDIR llvm
    LIST_SEPARATOR ","
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG llvmorg-18.1.0
    GIT_PROGRESS ON
    DEPENDS python-venv
)

add_custom_target(llvm-lit ALL COMMAND cp ${TTMLIR_TOOLCHAIN}/src/llvm-project-build/bin/llvm-lit ${TTMLIR_TOOLCHAIN}/bin/llvm-lit DEPENDS llvm-project)
