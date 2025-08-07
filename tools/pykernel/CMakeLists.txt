include(AddMLIRPython)

set(TTPYKERNEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

declare_mlir_python_sources(TTPykernelSources
    ROOT_DIR "${TTPYKERNEL_ROOT_DIR}"
    SOURCES
        ast.py
        op.py
        types.py
)

add_mlir_python_modules(TTPykernelModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/pykernel"
    INSTALL_PREFIX "python_packages/pykernel"
    DECLARED_SOURCES TTPykernelSources
)


# Include the Logic Here to install PyKernel dependencies (to run)

find_program(GREP_EXEC grep REQUIRED)
set(GREP_PATTERN [[^(git\+|^-r|^#)]])
set(METAL_REQUIREMENTS_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/python_env/requirements-dev.txt)

# Set extra build dependencies here that might be filtered out from the requirements-dev.txt
set(PYKERNEL_EXTRA_PIP_PACKAGES graphviz)

add_custom_command(TARGET TTPykernelModules POST_BUILD
    COMMENT "Setting up PyKernel dependencies"
    COMMAND ${GREP_EXEC} -vE '${GREP_PATTERN}' ${METAL_REQUIREMENTS_PATH} > ${CMAKE_CURRENT_BINARY_DIR}/pykernel_requirements.txt
    COMMAND PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -r ${CMAKE_CURRENT_BINARY_DIR}/pykernel_requirements.txt
    COMMAND PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install ${PYKERNEL_EXTRA_PIP_PACKAGES}
)

# Add dependency to ensure tt-metal is downloaded before the command runs
add_dependencies(TTPykernelModules tt-metal-download)
