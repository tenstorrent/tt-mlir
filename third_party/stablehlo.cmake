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

include(stablehlo_libs.cmake)
foreach(lib_name lib_path IN ZIP_LISTS TTMLIR_STABLEHLO_LIBRARY_NAMES TTMLIR_STABLEHLO_LIBRARIES)
  add_library(${lib_name} SHARED IMPORTED GLOBAL)
  set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${lib_path})
  add_dependencies(${lib_name} stablehlo)
endforeach()

