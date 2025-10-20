add_subdirectory(ttmlir-opt)
add_subdirectory(ttmlir-lsp-server)

if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(builder)
  add_subdirectory(goldens)

  if (TTMLIR_ENABLE_PYKERNEL)
    add_subdirectory(pykernel)
    add_subdirectory(ttnn-jit)
  endif()

  if (TTMLIR_ENABLE_EXPLORER)
    add_subdirectory(explorer)
  endif()

  add_subdirectory(ttrt)
  add_subdirectory(op-by-op-infra)
endif()

add_subdirectory(ttmlir-translate)
add_subdirectory(ttnn-standalone)
add_subdirectory(tt-alchemist)
