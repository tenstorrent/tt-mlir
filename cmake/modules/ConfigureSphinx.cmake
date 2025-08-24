# Configure paths
set(SPHINX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/docs/sphinx)
set(SPHINX_BUILD_DIR ${CMAKE_BINARY_DIR}/docs/book/sphinx)
set(SPHINX_MD_DIR ${CMAKE_BINARY_DIR}/docs/src/autogen/md/Module)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${SPHINX_BUILD_DIR})
file(MAKE_DIRECTORY ${SPHINX_MD_DIR})
file(MAKE_DIRECTORY ${SPHINX_SOURCE_DIR}/generated)
