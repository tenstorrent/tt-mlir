# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vwells/sources/tt-mlir

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vwells/sources/tt-mlir

# Utility rule file for install-TTMLIRTTIRToTTNN.

# Include any custom commands dependencies for this target.
include lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/progress.make

lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN:
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TTIRToTTNN && /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="TTMLIRTTIRToTTNN" -P /home/vwells/sources/tt-mlir/cmake_install.cmake

lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/codegen:
.PHONY : lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/codegen

install-TTMLIRTTIRToTTNN: lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN
install-TTMLIRTTIRToTTNN: lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/build.make
.PHONY : install-TTMLIRTTIRToTTNN

# Rule to build all files generated by this target.
lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/build: install-TTMLIRTTIRToTTNN
.PHONY : lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/build

lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Conversion/TTIRToTTNN && $(CMAKE_COMMAND) -P CMakeFiles/install-TTMLIRTTIRToTTNN.dir/cmake_clean.cmake
.PHONY : lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/clean

lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Conversion/TTIRToTTNN /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Conversion/TTIRToTTNN /home/vwells/sources/tt-mlir/lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Conversion/TTIRToTTNN/CMakeFiles/install-TTMLIRTTIRToTTNN.dir/depend
