set(LIB_NAME TTNNOpModelLib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    TTNNOpModelLib.cpp
    SingletonDeviceContext.cpp
)
add_library(${LIB_NAME} STATIC ${SOURCES})

message(STATUS "TTMLIR_ENABLE_OP_MODEL[${TTMLIR_ENABLE_OP_MODEL}]")
if (TTMLIR_ENABLE_OPMODEL)
    # Link to tt-metal libs and include directories
    target_include_directories(${LIB_NAME} PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
    target_link_libraries(${LIB_NAME} PUBLIC TTNN_LIBRARY TTMETAL_LIBRARY DEVICE_LIBRARY)
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


# Add TTNNOpModelLib to the export set
install(TARGETS ${LIB_NAME}
        EXPORT TTNNOpModelLibTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Export the targets
export(EXPORT TTNNOpModelLibTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/TTNNOpModelLibTargets.cmake"
       NAMESPACE TTNN::)
