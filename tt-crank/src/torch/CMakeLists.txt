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
        "TT_CRANK_BUILD_PYTHON_EXT=ON but `import torch` failed in ${Python_EXECUTABLE}. "
        "Install torch in the venv first (`source venv/activate` covers this).")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tt_crank_torch_native NB_STATIC
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

target_compile_definitions(_tt_crank_torch_native PRIVATE
    TT_CRANK_SIM_DIR="${TT_CRANK_SIM_DIR}"
    TT_CRANK_TT_METAL_HOME="${TT_CRANK_TT_METAL_HOME}"
)

# The SoC descriptor under TT_CRANK_SIM_DIR is copied at build time; ensure
# that's run before _native.so can be loaded against ttsim.
add_dependencies(_tt_crank_torch_native tt_crank_ttsim_stage)

# Default RTTI here — torch headers expect it. tt_crank's -fno-rtti is PRIVATE
# (see note in src/CMakeLists.txt) so it doesn't transitively force consumers.
target_link_libraries(_tt_crank_torch_native PRIVATE
    tt_crank::tt_crank
    tt_crank_common
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
target_link_libraries(_tt_crank_torch_native PRIVATE ${TORCH_PYTHON_LIBRARY})

# Final import name on disk is `_native.so` so Python sees tt_crank.torch._native.
# INSTALL_RPATH is for the wheel, where the installed _native.so lives in
# tt_crank/torch/: bundled native libs are one level up in tt_crank/lib
# ($ORIGIN/../lib); libtorch is NOT bundled (it's a declared dependency) and
# resolves from the sibling torch wheel at $ORIGIN/../../torch/lib. The editable
# install uses the build-tree module, which keeps CMake's build RPATH.
set_target_properties(_tt_crank_torch_native PROPERTIES
    OUTPUT_NAME _native
    PREFIX ""
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../torch/lib"
)
