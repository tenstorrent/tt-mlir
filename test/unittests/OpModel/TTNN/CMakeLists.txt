if (TTMLIR_ENABLE_TTNN_PY)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

add_subdirectory(Conversion)
add_subdirectory(Lib)
add_subdirectory(Op)
