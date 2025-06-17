add_subdirectory(ttmlir-opt)
add_subdirectory(ttmlir-lsp-server)
if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  if (TTMLIR_ENABLE_TTIRBUILDER)
    add_subdirectory(ttir_builder)
  endif()

  if (TTMLIR_ENABLE_PYKERNEL)
    add_subdirectory(pykernel)
  endif()

  if (TTMLIR_ENABLE_EXPLORER)
    add_subdirectory(explorer)
  endif()

endif()
add_subdirectory(ttmlir-translate)

add_subdirectory(ttnn-standalone)
