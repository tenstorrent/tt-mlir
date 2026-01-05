add_subdirectory(ttmlir-opt)
add_subdirectory(ttmlir-lsp-server)

if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(builder)
  add_subdirectory(golden)
  add_subdirectory(profiler)

  if (TTMLIR_ENABLE_PYKERNEL)
    add_subdirectory(pykernel)
  endif()

  if (TTMLIR_ENABLE_TTNN_JIT)
    add_subdirectory(ttnn-jit)
  endif()

  if (TTMLIR_ENABLE_EXPLORER)
    add_subdirectory(explorer)
  endif()

  if (TTMLIR_ENABLE_TTRT)
    add_subdirectory(ttrt)
  endif()
  add_subdirectory(op-by-op-infra)
endif()

add_subdirectory(ttmlir-translate)
add_subdirectory(ttnn-standalone)

# TTMLIR_ENABLE_RUNTIME is needed because tt-metal doesn't build without it => can't install TT-NN dependencies
if (TTMLIR_ENABLE_ALCHEMIST AND TTMLIR_ENABLE_RUNTIME)
  add_subdirectory(tt-alchemist)
endif()
