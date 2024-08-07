include(ExternalProject)

set(STABLE_HLO_VERSION "v1.0.0")

ExternalProject_Add(
    stablehlo
    PREFIX ${TTMLIR_SOURCE_DIR}/third_party/stablehlo
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
      # TODO: Figure out if we need this (or benefit from it) when it comes to linking stablehlo.
      #-DSTABLEHLO_BUILD_EMBEDDED=ON
      -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
    GIT_REPOSITORY https://github.com/openxla/stablehlo
    GIT_TAG ${STABLE_HLO_VERSION}
    GIT_PROGRESS ON
)