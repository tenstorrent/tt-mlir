set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_mlir_unittest(ThreeWayConsistencyTests
    ThreeWayConsistency.cpp
    PARTIAL_SOURCES_INTENDED
)

target_compile_options(ThreeWayConsistencyTests
    PRIVATE
    -frtti
    -fexceptions
)

target_include_directories(ThreeWayConsistencyTests
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${PROJECT_SOURCE_DIR}/runtime/include/
    ${PROJECT_BINARY_DIR}/runtime/include/
    ${PROJECT_BINARY_DIR}/runtime/include/tt/runtime/flatbuffer/
)

target_link_libraries(ThreeWayConsistencyTests
    PRIVATE
    gtest
    gtest_main
    MLIRTTCoreDialect
    MLIRTTNNDialect
    TTRuntimeTTNN
)


message(STATUS "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
