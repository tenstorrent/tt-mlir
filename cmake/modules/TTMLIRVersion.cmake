# get git commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE TTMLIR_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# get the latest tag from Git
execute_process(
  COMMAND git describe --tags --abbrev=0
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# get the number of commits since the latest tag
execute_process(
  COMMAND git rev-list ${GIT_TAG}..HEAD --count
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_COMMITS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract the major and minor version from the tag (assumes tags are in "major.minor" format)
string(REGEX MATCH "^v([0-9]+)\\.([0-9]+)$" GIT_TAG_MATCH ${GIT_TAG})
set(TTMLIR_VERSION_MAJOR ${CMAKE_MATCH_1})
set(TTMLIR_VERSION_MINOR ${CMAKE_MATCH_2})
set(TTMLIR_VERSION_PATCH ${GIT_COMMITS})

message(STATUS "Project commit hash: ${TTMLIR_GIT_HASH}")
message(STATUS "Project version: ${TTMLIR_VERSION_MAJOR}.${TTMLIR_VERSION_MINOR}.${TTMLIR_VERSION_PATCH}")

add_definitions("-DTTMLIR_GIT_HASH=${TTMLIR_GIT_HASH}")
add_definitions("-DTTMLIR_VERSION_MAJOR=${TTMLIR_VERSION_MAJOR}")
add_definitions("-DTTMLIR_VERSION_MINOR=${TTMLIR_VERSION_MINOR}")
add_definitions("-DTTMLIR_VERSION_PATCH=${TTMLIR_VERSION_PATCH}")
