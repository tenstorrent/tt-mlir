#!/bin/bash

# Common functions for git bisect scripts

# Exit code constants for log_result function
UP_SKIP=125           # Skip commit as untestable
UP_BAD=1              # Mark commit as bad

# Function to log command results and return appropriate exit code for git bisect
# Usage: log_result <exit_code> [cmd_name] [failure_exit_code]
# exit_code: exit code from the command
# cmd_name: optional command name, defaults to "cmd"
# failure_exit_code: optional exit code to use on failure, defaults to UP_BAD
#   - UP_SKIP (125): skip untestable commit
#   - UP_BAD (1): mark commit as bad
# Returns: exit code 0 if command succeeded, otherwise uses failure_exit_code
log_result() {
    local exit_code=$1
    local cmd_name=${2:-"cmd"}
    local failure_exit_code=${3:-$UP_BAD}

    if [ $exit_code -eq 0 ]; then
        echo -e "${cmd_name} succeeded\n" >> logs/bisect_log.log
    else
        echo -e "${cmd_name} failed with exit code ${exit_code}\n" >> logs/bisect_log.log
        exit $failure_exit_code
    fi
}
