#credit: https://github.com/google/libcxx/blob/master/cmake/Modules/GetTriple.cmake
# Get the architecture.
set(arch ${CMAKE_SYSTEM_PROCESSOR})
if (arch STREQUAL "x86")
  set(arch "i686")
endif()
# Get the vendor.
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(vendor "apple")
else()
  set(vendor "pc")
endif()
# Get os.
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(os "win32")
else()
  string(TOLOWER ${CMAKE_SYSTEM_NAME} os)
endif()
set(triple "${arch}-${vendor}-${os}")
set(${out} ${triple} PARENT_SCOPE)
set(${out_arch} ${arch} PARENT_SCOPE)
set(${out_vendor} ${vendor} PARENT_SCOPE)
set(${out_os} ${os} PARENT_SCOPE)
message(STATUS "Target triple: ${triple}")

add_definitions(-DTARGET_TRIPLE="${triple}")

if (NOT DEVICE_RUNTIME_ENABLED)
  add_library(TTRuntimeTypes INTERFACE)
  add_library(TTRuntimeUtils INTERFACE)
  add_library(TTRuntimeSysDesc INTERFACE)
  add_library(TTRuntimeContext INTERFACE)
  add_library(TTRuntimeSocket INTERFACE)
  add_library(TTRuntimeDebug INTERFACE)
  add_library(TTRuntimePerf INTERFACE)
  add_library(TTRuntimeWorkarounds INTERFACE)
  add_library(TTRuntimeDylibs INTERFACE)
  return()
endif()

add_library(TTRuntimeTypes STATIC types.cpp)
set_property(TARGET TTRuntimeTypes PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTypes
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeTypes PUBLIC coverage_config)
add_dependencies(TTRuntimeTypes RUNTIME_FBS_GENERATION)

add_library(TTRuntimeUtils STATIC utils.cpp)
set_property(TARGET TTRuntimeUtils PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeUtils
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeUtils PUBLIC coverage_config)
add_dependencies(TTRuntimeUtils RUNTIME_FBS_GENERATION)

add_library(TTRuntimeSysDesc STATIC system_desc.cpp)
set_property(TARGET TTRuntimeSysDesc PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeSysDesc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTRuntimeSysDesc RUNTIME_FBS_GENERATION)

target_include_directories(TTRuntimeSysDesc SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeSysDesc tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeSysDesc PUBLIC coverage_config)

add_library(TTRuntimeContext STATIC runtime_context.cpp)
set_property(TARGET TTRuntimeContext PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeContext
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeContext PUBLIC coverage_config)
add_dependencies(TTRuntimeContext RUNTIME_FBS_GENERATION)

add_library(TTRuntimeSocket STATIC socket.cpp)
set_property(TARGET TTRuntimeSocket PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeSocket
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeSocket PUBLIC coverage_config)
add_dependencies(TTRuntimeSocket RUNTIME_FBS_GENERATION)

add_library(TTRuntimeDebug STATIC debug.cpp)
set_property(TARGET TTRuntimeDebug PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDebug
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeDebug PUBLIC coverage_config)
add_dependencies(TTRuntimeDebug RUNTIME_FBS_GENERATION)

add_library(TTRuntimePerf STATIC perf.cpp)
set_property(TARGET TTRuntimePerf PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimePerf
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
)
add_dependencies(TTRuntimePerf tt-metal)
target_link_libraries(TTRuntimePerf PUBLIC coverage_config)

add_library(TTRuntimeWorkarounds STATIC workarounds.cpp)
set_property(TARGET TTRuntimeWorkarounds PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeWorkarounds
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeWorkarounds PUBLIC coverage_config)
add_dependencies(TTRuntimeWorkarounds RUNTIME_FBS_GENERATION)

add_library(TTRuntimeDylibs STATIC dylib.cpp)
set_property(TARGET TTRuntimeDylibs PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDylibs
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(TTRuntimeDylibs PUBLIC coverage_config)
add_dependencies(TTRuntimeDylibs RUNTIME_FBS_GENERATION)
