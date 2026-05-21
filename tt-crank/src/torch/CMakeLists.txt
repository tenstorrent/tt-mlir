find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Discover libtorch via the interpreter scikit-build-core handed us. This works
# whether torch is the venv's `pip install torch` or a vendored build.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
    "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX
    RESULT_VARIABLE _TORCH_RC
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _TORCH_RC EQUAL 0)
    message(FATAL_ERROR
        "TT_KURBLA_BUILD_PYTHON_EXT=ON but `import torch` failed in ${Python_EXECUTABLE}. "
        "Install torch in the venv first (requirements-dev.txt covers this).")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tt_kurbla_torch_native NB_STATIC
    _native.cpp
    backend.cpp
    ops/elementwise.cpp
    ops/fallback.cpp
    ops/tensor.cpp
    tensor.cpp
    ttir_module_builder.cpp
)

target_compile_definitions(_tt_kurbla_torch_native PRIVATE
    TT_KURBLA_SIM_DIR="${TT_KURBLA_SIM_DIR}"
    TT_KURBLA_TT_METAL_HOME="${TT_KURBLA_TT_METAL_HOME}"
)

# The SoC descriptor under TT_KURBLA_SIM_DIR is copied at build time; ensure
# that's run before _native.so can be loaded against ttsim.
add_dependencies(_tt_kurbla_torch_native tt_kurbla_ttsim_stage)

# Default RTTI here — torch headers expect it. tt_kurbla's -fno-rtti is PRIVATE
# (see note in src/CMakeLists.txt) so it doesn't transitively force consumers.
target_link_libraries(_tt_kurbla_torch_native PRIVATE
    tt_kurbla::tt_kurbla
    tt_kurbla_common
    ${TORCH_LIBRARIES}
)

# Final import name on disk is `_native.so` so Python sees tt_kurbla.torch._native.
# INSTALL_RPATH points at the in-build-tree locations of libtt_kurbla.so,
# tt-mlir's shared libs, and torch's runtime so the .so can dlopen them from
# site-packages without LD_LIBRARY_PATH. Editable-install only; a packaged
# wheel would bundle and use $ORIGIN instead.
set_target_properties(_tt_kurbla_torch_native PROPERTIES
    OUTPUT_NAME _native
    PREFIX ""
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/src;${CMAKE_BINARY_DIR}/src/common;${CMAKE_BINARY_DIR}/third_party/tt-mlir-install/lib;${TORCH_INSTALL_PREFIX}/lib"
)

install(TARGETS _tt_kurbla_torch_native
    LIBRARY DESTINATION tt_kurbla/torch
)
