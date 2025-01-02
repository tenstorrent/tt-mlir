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
include lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/flags.make

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/codegen:
.PHONY : lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/codegen

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/flags.make
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o: lib/Dialect/TTIR/IR/TTIRDialect.cpp
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o -MF CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o.d -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIRDialect.cpp

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIRDialect.cpp > CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.i

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIRDialect.cpp -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.s

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/flags.make
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o: lib/Dialect/TTIR/IR/TTIROps.cpp
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o -MF CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o.d -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROps.cpp

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROps.cpp > CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.i

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROps.cpp -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.s

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/flags.make
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o: lib/Dialect/TTIR/IR/TTIROpsInterfaces.cpp
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o -MF CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o.d -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROpsInterfaces.cpp

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROpsInterfaces.cpp > CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.i

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/TTIROpsInterfaces.cpp -o CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.s

obj.MLIRTTIRDialect: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIRDialect.cpp.o
obj.MLIRTTIRDialect: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROps.cpp.o
obj.MLIRTTIRDialect: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/TTIROpsInterfaces.cpp.o
obj.MLIRTTIRDialect: lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/build.make
.PHONY : obj.MLIRTTIRDialect

# Rule to build all files generated by this target.
lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/build: obj.MLIRTTIRDialect
.PHONY : lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/build

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR && $(CMAKE_COMMAND) -P CMakeFiles/obj.MLIRTTIRDialect.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/clean

lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TTIR/IR/CMakeFiles/obj.MLIRTTIRDialect.dir/depend
