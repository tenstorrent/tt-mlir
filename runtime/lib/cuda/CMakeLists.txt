if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_CUDA)
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
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/GPU
   ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(TTRuntimeCUDA PUBLIC flatbuffers)
target_link_libraries(TTRuntimeCUDA PUBLIC coverage_config)
target_link_libraries(TTRuntimeCUDA PUBLIC LLVMSupport)
target_link_libraries(TTRuntimeCUDA PUBLIC CUDA::cudart)
target_link_libraries(TTRuntimeCUDA PUBLIC CUDA::cuda_driver)

add_dependencies(TTRuntimeCUDA FBS_GENERATION)

if (CUDA_RUNTIME_ENABLED)
  add_executable(test_cuda_program_executor test_program_executor.cpp)
  set_property(TARGET test_cuda_program_executor PROPERTY CXX_STANDARD 20)
  target_link_libraries(test_cuda_program_executor PRIVATE TTRuntimeCUDA TTBinary)
  target_include_directories(test_cuda_program_executor PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/GPU
  )
  target_link_directories(test_cuda_program_executor PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
  target_link_directories(test_cuda_program_executor PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib64)
  add_dependencies(test_cuda_program_executor TTRuntimeCUDA TTBinary FBS_GENERATION)
endif()
