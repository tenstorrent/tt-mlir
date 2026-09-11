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
        "Install torch in the venv first (`source venv/activate` covers this).")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tt_kurbla_torch_native NB_STATIC
    _native.cpp
    backend.cpp
    ops/conv.cpp
    ops/elementwise.cpp
    ops/fallback.cpp
    ops/linear.cpp
    ops/sdpa.cpp
    ops/tensor.cpp
    tensor.cpp
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

# torch_python ships separately from libtorch_cpu/c10 and isn't pulled in by
# ${TORCH_LIBRARIES}. The torch.compile path's _native.cpp uses THPVariable_*
# to unpack/wrap torch.Tensors across the nanobind boundary; those symbols
# (e.g. ParameterClass) live in libtorch_python.so.
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH REQUIRED
)
target_link_libraries(_tt_kurbla_torch_native PRIVATE ${TORCH_PYTHON_LIBRARY})

# Final import name on disk is `_native.so` so Python sees tt_kurbla.torch._native.
# INSTALL_RPATH lets the .so dlopen libtt_kurbla.so, tt-mlir's shared libs, and
# torch's runtime from site-packages without LD_LIBRARY_PATH.
#   - Editable/dev: point at the in-build-tree locations.
#   - Wheel: _native.so lives in tt_kurbla/torch/. Bundled native libs are one
#     level up in tt_kurbla/lib ($ORIGIN/../lib); libtorch is NOT bundled (it's a
#     declared dependency) and resolves from the sibling torch wheel at
#     $ORIGIN/../../torch/lib for a flat site-packages layout.
if(TT_KURBLA_WHEEL)
    set(_native_install_rpath "$ORIGIN/../lib;$ORIGIN/../../torch/lib")
else()
    set(_native_install_rpath "${CMAKE_BINARY_DIR}/src;${CMAKE_BINARY_DIR}/src/common;${CMAKE_BINARY_DIR}/third_party/tt-mlir-install/lib;${TORCH_INSTALL_PREFIX}/lib")
endif()
# For the wheel, bake the $ORIGIN rpath at link time (BUILD_WITH_INSTALL_RPATH)
# for the same reason as the other bundled libs. Editable keeps the default
# (build with build-tree rpath, rewrite to INSTALL_RPATH at install).
set_target_properties(_tt_kurbla_torch_native PROPERTIES
    OUTPUT_NAME _native
    PREFIX ""
    INSTALL_RPATH "${_native_install_rpath}"
    BUILD_WITH_INSTALL_RPATH ${TT_KURBLA_WHEEL}
)

install(TARGETS _tt_kurbla_torch_native
    LIBRARY
        DESTINATION tt_kurbla/torch
        COMPONENT wheel_payload
)
