# credit: https://github.com/google/libcxx/blob/master/cmake/Modules/GetTriple.cmake
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

# Construct the target triple
set(TARGET_TRIPLE "triple")

add_definitions(-DTARGET_TRIPLE="${TARGET_TRIPLE}")

add_library(TTRuntimeSysDesc STATIC system_desc.cpp)
set_property(TARGET TTRuntimeSysDesc PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeSysDesc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeSysDesc PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeSysDesc tt-metal FBS_GENERATION)

add_library(TTRuntimeDebug STATIC debug.cpp)
set_property(TARGET TTRuntimeDebug PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeDebug
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)

add_library(TTRuntimeWorkarounds STATIC workarounds.cpp)
set_property(TARGET TTRuntimeWorkarounds PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeWorkarounds
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
)
