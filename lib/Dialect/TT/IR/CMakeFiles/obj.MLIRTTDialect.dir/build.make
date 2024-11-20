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
include lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/flags.make

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/codegen:
.PHONY : lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/codegen

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/flags.make
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o: lib/Dialect/TT/IR/TTOpsTypes.cpp
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o -MF CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o.d -o CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp > CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.i

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp -o CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.s

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/flags.make
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o: lib/Dialect/TT/IR/TTDialect.cpp
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o -MF CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o.d -o CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTDialect.cpp

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTDialect.cpp > CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.i

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTDialect.cpp -o CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.s

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/flags.make
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o: lib/Dialect/TT/IR/TTOps.cpp
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o -MF CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o.d -o CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOps.cpp

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOps.cpp > CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.i

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/TTOps.cpp -o CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.s

obj.MLIRTTDialect: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o
obj.MLIRTTDialect: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o
obj.MLIRTTDialect: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o
obj.MLIRTTDialect: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/build.make
.PHONY : obj.MLIRTTDialect

# Rule to build all files generated by this target.
lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/build: obj.MLIRTTDialect
.PHONY : lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/build

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && $(CMAKE_COMMAND) -P CMakeFiles/obj.MLIRTTDialect.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/clean

lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/depend

