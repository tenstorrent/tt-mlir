include(AddMLIRPython)

set(TTNNJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

declare_mlir_python_sources(TTNNJITSources
    ROOT_DIR "${TTNNJIT_ROOT_DIR}"
    SOURCES
        api.py
        __init__.py
        _src/ttir_ast.py
        _src/utils.py
)

add_mlir_python_modules(TTNNJITModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttnn_jit"
    INSTALL_PREFIX "python_packages/ttnn_jit"
    DECLARED_SOURCES TTNNJITSources
)

add_custom_target(ttnn-jit COMMENT "ttnn-jit" DEPENDS TTNNJITModules)
add_dependencies(ttnn-jit tt-metal-download)

# Install ttnn-jit runtime dependencies (ttnn pip packages)
find_program(GREP_EXEC grep REQUIRED)
set(GREP_PATTERN [[^(git\+|^-r|^#)]])
set(METAL_REQUIREMENTS_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/python_env/requirements-dev.txt)

# Set extra build dependencies here that depend on the pip install of ttnn
set(TTNNJIT_EXTRA_PIP_PACKAGES graphviz)

add_custom_command(TARGET ttnn-jit POST_BUILD
    COMMENT "Setting up ttnn-jit dependencies"
    COMMAND ${GREP_EXEC} -vE '${GREP_PATTERN}' ${METAL_REQUIREMENTS_PATH} > ${CMAKE_CURRENT_BINARY_DIR}/ttnn_jit_requirements.txt
    COMMAND PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -r ${CMAKE_CURRENT_BINARY_DIR}/ttnn_jit_requirements.txt
    COMMAND PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install ${TTNNJIT_EXTRA_PIP_PACKAGES}
)
