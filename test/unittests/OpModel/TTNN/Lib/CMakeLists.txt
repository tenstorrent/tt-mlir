if (TTMLIR_ENABLE_OPMODEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TestOpModelLib
TestOpModelLib.cpp
)

target_include_directories(TestOpModelLib
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
)

target_link_libraries(TestOpModelLib
    PRIVATE
    gtest
    TTNNOpModelLib
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
)
endif()