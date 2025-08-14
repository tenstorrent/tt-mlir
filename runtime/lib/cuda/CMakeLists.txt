if (TT_RUNTIME_ENABLE_CUDA)
  set(CUDA_RUNTIME_ENABLED ON)
else()
  set(CUDA_RUNTIME_ENABLED OFF)
endif()

if (NOT CUDA_RUNTIME_ENABLED)
  add_library(TTRuntimeCUDA INTERFACE)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

add_library(TTRuntimeCUDA
  STATIC
  program_executor.cpp
)




set_property(TARGET TTRuntimeCUDA PROPERTY CXX_STANDARD 20)
target_compile_definitions(TTRuntimeCUDA PUBLIC TT_RUNTIME_ENABLE_CUDA)

target_include_directories(TTRuntimeCUDA PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/CUDA
   ${CUDA_INCLUDE_DIRS}
)

find_library(FLATBUFFERS_LIB flatbuffers PATHS ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_libraries(TTRuntimeCUDA PUBLIC ${FLATBUFFERS_LIB})
target_link_libraries(TTRuntimeCUDA PUBLIC coverage_config)
target_link_libraries(TTRuntimeCUDA PUBLIC LLVMSupport)
target_link_libraries(TTRuntimeCUDA PUBLIC CUDA::cudart)
target_link_libraries(TTRuntimeCUDA PUBLIC CUDA::cuda_driver)

add_dependencies(TTRuntimeCUDA FBS_GENERATION)
