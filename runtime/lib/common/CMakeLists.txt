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

if (TTMLIR_ENABLE_RUNTIME AND (TT_RUNTIME_ENABLE_TTNN OR TT_RUNTIME_ENABLE_TTMETAL))
    set(ANY_RUNTIME_ENABLED ON)
else()
    set(ANY_RUNTIME_ENABLED OFF)
endif()

if (NOT ANY_RUNTIME_ENABLED)
  add_library(TTRuntimeSysDesc INTERFACE)
  add_library(TTRuntimeContext INTERFACE)
  add_library(TTRuntimeDebug INTERFACE)
  add_library(TTRuntimePerf INTERFACE)
  add_library(TTRuntimeWorkarounds INTERFACE)
  add_library(TTRuntimeDylibs INTERFACE)
  return()
endif()

add_library(TTRuntimeSysDesc STATIC system_desc.cpp)
set_property(TARGET TTRuntimeSysDesc PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeSysDesc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_include_directories(TTRuntimeSysDesc SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeSysDesc tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeSysDesc PUBLIC coverage_config)

add_library(TTRuntimeContext STATIC runtime_context.cpp)
set_property(TARGET TTRuntimeContext PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeContext
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(TTRuntimeContext PUBLIC coverage_config)

add_library(TTRuntimeDebug STATIC debug.cpp)
set_property(TARGET TTRuntimeDebug PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDebug
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(TTRuntimeDebug PUBLIC coverage_config)

add_library(TTRuntimePerf STATIC perf.cpp)
set_property(TARGET TTRuntimePerf PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimePerf
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
)
add_dependencies(TTRuntimePerf tt-metal)
target_link_libraries(TTRuntimePerf PUBLIC coverage_config)

add_library(TTRuntimeWorkarounds STATIC workarounds.cpp)
set_property(TARGET TTRuntimeWorkarounds PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeWorkarounds
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(TTRuntimeWorkarounds PUBLIC coverage_config)

add_library(TTRuntimeDylibs STATIC dylib.cpp)
set_property(TARGET TTRuntimeDylibs PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDylibs
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(TTRuntimeDylibs PUBLIC coverage_config)
