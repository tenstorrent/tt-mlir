
file(GLOB TTMLIR_STABLEHLO_LIBRARIES "${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build/lib/*.a")
set(TTMLIR_STABLEHLO_LIBRARY_NAMES)
foreach(TTMLIR_STABLEHLO_LIBRARY ${TTMLIR_STABLEHLO_LIBRARIES})
    get_filename_component(lib_name ${TTMLIR_STABLEHLO_LIBRARY} NAME_WE)
    string(REPLACE "lib" "" lib_name ${lib_name}) # Remove the "lib" prefix if it exists
    list(APPEND TTMLIR_STABLEHLO_LIBRARY_NAMES ${lib_name})
endforeach()
