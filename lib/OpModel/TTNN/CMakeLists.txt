set(LIB_NAME TTNNOpModelLib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    TTNNOpModel.cpp
    Conversion.cpp
    SingletonDeviceContext.cpp
)
add_library(${LIB_NAME} STATIC ${SOURCES})

message(STATUS "TTMLIR_ENABLE_OPMODEL[${TTMLIR_ENABLE_OPMODEL}]")
if (TTMLIR_ENABLE_OPMODEL)
    # Building op model library will invoke building tt-metal; and it requires TT_METAL_HOME environment variable to be set.
    if("$ENV{TT_METAL_HOME}" STREQUAL "")
        message(WARNING "TT_METAL_HOME is not set")
    endif()

    # Link to tt-metal libs and include directories
    target_link_libraries(${LIB_NAME} PUBLIC TTNN::CPP TT::Metalium)
    target_compile_definitions(${LIB_NAME} PUBLIC TTMLIR_ENABLE_OPMODEL)
else()
    # link stubs implementation when op model library is disabled
    message(WARNING "TTNNOpModelLib is disabled. The optimizer will not achieve optimal performance.")
endif()

# Specify the include directories for the library
target_include_directories(${LIB_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/include/ttmlir/OpModel/TTNN/)

target_link_libraries(${LIB_NAME} PUBLIC coverage_config)
