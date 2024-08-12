include(ExternalProject)

set(STABLE_HLO_VERSION "v1.0.0")

ExternalProject_Add(
    stablehlo
    PREFIX ${PROJECT_SOURCE_DIR}/third_party/stablehlo
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      # TODO: Figure out if we need this (or benefit from it) when it comes to linking stablehlo.
      #-DSTABLEHLO_BUILD_EMBEDDED=ON
      -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
    GIT_REPOSITORY https://github.com/openxla/stablehlo
    GIT_TAG ${STABLE_HLO_VERSION}
    GIT_PROGRESS ON
)

file(GLOB TTMLIR_STABLEHLO_LIBRARIES "${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib/*.a")
foreach(TTMLIR_STABLEHLO_LIBRARY ${TTMLIR_STABLEHLO_LIBRARIES})
    get_filename_component(lib_name ${TTMLIR_STABLEHLO_LIBRARY} NAME_WE)
    string(REPLACE "lib" "" lib_name ${lib_name}) # Remove the "lib" prefix if it exists
    message(STATUS "Adding TTMLIR library: ${lib_name}")
    add_library(${lib_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TTMLIR_STABLEHLO_LIBRARY})
    add_dependencies(${lib_name} stablehlo)
    list(APPEND STABLEHLO_LIBRARIES_LIST ${lib_name})
endforeach()

set_property(GLOBAL PROPERTY STABLEHLO_LIBS "${STABLEHLO_LIBRARIES_LIST}")

