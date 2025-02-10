# get git commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE TTMLIR_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND bash "-c" "git tag --merged main --sort=-taggerdate"
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE VLAD_TRACE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(TODO_remove_this_VLAD_TRACE_is "[${VLAD_TRACE}]")

# get the latest tag from git, reachable from 'main' branch and matching 'v<major>.<minor>' format
execute_process(
  COMMAND bash "-c" "git tag --merged main --sort=-taggerdate | egrep '^v([0-9]+)\.([0-9]+)$' | head -1"
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(TODO_remove_this_GIT_TAG_is "[${GIT_TAG}]")

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
