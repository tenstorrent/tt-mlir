# Find Sphinx executables
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Sphinx documentation generator"
)

find_program(SPHINX_APIDOC_EXECUTABLE
    NAMES sphinx-apidoc
    DOC "Sphinx API doc generator"
)

if(NOT SPHINX_EXECUTABLE)
    message(FATAL_ERROR "sphinx-build not found - required to build documentation")
endif()

if(NOT SPHINX_APIDOC_EXECUTABLE)
    message(FATAL_ERROR "sphinx-apidoc not found - required to build API documentation")
endif()

# Configure paths
set(SPHINX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/docs/source)
set(SPHINX_BUILD_DIR ${CMAKE_BINARY_DIR}/docs/book/sphinx)
set(SPHINX_MD_DIR ${CMAKE_BINARY_DIR}/docs/src/autogen/md/Module)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${SPHINX_BUILD_DIR})
file(MAKE_DIRECTORY ${SPHINX_MD_DIR})
file(MAKE_DIRECTORY ${SPHINX_SOURCE_DIR}/generated)

# Configure sphinx-apidoc command
set(SPHINX_APIDOC_COMMAND ${SPHINX_APIDOC_EXECUTABLE}
    -o ${SPHINX_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/python_packages/ttir_builder
)

set(SPHINX_BUILD_COMMAND ${SPHINX_EXECUTABLE}
    -b markdown
    ${SPHINX_SOURCE_DIR}
    ${SPHINX_MD_DIR}
)
