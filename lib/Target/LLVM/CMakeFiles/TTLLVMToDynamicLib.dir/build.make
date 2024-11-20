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

# Include any dependencies generated for this target.
include lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/flags.make

lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/codegen:
.PHONY : lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/codegen

# Object files for target TTLLVMToDynamicLib
TTLLVMToDynamicLib_OBJECTS =

# External object files for target TTLLVMToDynamicLib
TTLLVMToDynamicLib_EXTERNAL_OBJECTS = \
"/home/vwells/sources/tt-mlir/lib/Target/LLVM/CMakeFiles/obj.TTLLVMToDynamicLib.dir/LLVMToDynamicLib.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Target/LLVM/CMakeFiles/obj.TTLLVMToDynamicLib.dir/LLVMToDynamicLibRegistration.cpp.o"

lib/libTTLLVMToDynamicLib.a: lib/Target/LLVM/CMakeFiles/obj.TTLLVMToDynamicLib.dir/LLVMToDynamicLib.cpp.o
lib/libTTLLVMToDynamicLib.a: lib/Target/LLVM/CMakeFiles/obj.TTLLVMToDynamicLib.dir/LLVMToDynamicLibRegistration.cpp.o
lib/libTTLLVMToDynamicLib.a: lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/build.make
lib/libTTLLVMToDynamicLib.a: lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../libTTLLVMToDynamicLib.a"
	cd /home/vwells/sources/tt-mlir/lib/Target/LLVM && $(CMAKE_COMMAND) -P CMakeFiles/TTLLVMToDynamicLib.dir/cmake_clean_target.cmake
	cd /home/vwells/sources/tt-mlir/lib/Target/LLVM && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TTLLVMToDynamicLib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/build: lib/libTTLLVMToDynamicLib.a
.PHONY : lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/build

lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Target/LLVM && $(CMAKE_COMMAND) -P CMakeFiles/TTLLVMToDynamicLib.dir/cmake_clean.cmake
.PHONY : lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/clean

lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Target/LLVM /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Target/LLVM /home/vwells/sources/tt-mlir/lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Target/LLVM/CMakeFiles/TTLLVMToDynamicLib.dir/depend

