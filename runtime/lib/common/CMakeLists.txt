# Get the target architecture
set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})

# Get the target vendor
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(TARGET_VENDOR "gnu")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(TARGET_VENDOR "apple")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(TARGET_VENDOR "microsoft")
endif()

# Get the target OS
set(TARGET_OS ${CMAKE_SYSTEM_NAME})

# Construct the target triple
set(TARGET_TRIPLE "${TARGET_ARCH}-${TARGET_VENDOR}-${TARGET_OS}")

add_definitions(-DTARGET_TRIPLE=${TARGET_TRIPLE})

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
