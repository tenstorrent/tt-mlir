if (NOT TTMLIR_ENABLE_CUDA)
  add_library(TTRuntimeCUDA INTERFACE)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

add_library(TTRuntimeCUDA
  STATIC
  program_executor.cpp
  runtime.cpp
)




set_property(TARGET TTRuntimeCUDA PROPERTY CXX_STANDARD 20)
target_compile_definitions(TTRuntimeCUDA PUBLIC TTMLIR_ENABLE_CUDA)

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
