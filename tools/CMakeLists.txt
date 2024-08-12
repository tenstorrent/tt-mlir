include(ExternalProject)

include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo)
include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build)

add_subdirectory(ttmlir-opt)
add_subdirectory(ttmlir-translate)
