# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/demo_ByteTrack.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo_ByteTrack.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo_ByteTrack.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo_ByteTrack.dir/flags.make

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/BYTETracker.cpp
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/BYTETracker.cpp

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/BYTETracker.cpp > CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.i

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/BYTETracker.cpp -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.s

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/STrack.cpp
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/STrack.cpp

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/STrack.cpp > CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.i

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/STrack.cpp -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.s

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/kalmanFilter.cpp
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/kalmanFilter.cpp

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/kalmanFilter.cpp > CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.i

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/kalmanFilter.cpp -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.s

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/lapjv.cpp
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/lapjv.cpp

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/lapjv.cpp > CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.i

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/lapjv.cpp -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.s

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/utils.cpp
CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/utils.cpp

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/utils.cpp > CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.i

CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/bytetrack/src/utils.cpp -o CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.s

CMakeFiles/demo_ByteTrack.dir/main.cpp.o: CMakeFiles/demo_ByteTrack.dir/flags.make
CMakeFiles/demo_ByteTrack.dir/main.cpp.o: /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/main.cpp
CMakeFiles/demo_ByteTrack.dir/main.cpp.o: CMakeFiles/demo_ByteTrack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/demo_ByteTrack.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_ByteTrack.dir/main.cpp.o -MF CMakeFiles/demo_ByteTrack.dir/main.cpp.o.d -o CMakeFiles/demo_ByteTrack.dir/main.cpp.o -c /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/main.cpp

CMakeFiles/demo_ByteTrack.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_ByteTrack.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/main.cpp > CMakeFiles/demo_ByteTrack.dir/main.cpp.i

CMakeFiles/demo_ByteTrack.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_ByteTrack.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/main.cpp -o CMakeFiles/demo_ByteTrack.dir/main.cpp.s

# Object files for target demo_ByteTrack
demo_ByteTrack_OBJECTS = \
"CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o" \
"CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o" \
"CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o" \
"CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o" \
"CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o" \
"CMakeFiles/demo_ByteTrack.dir/main.cpp.o"

# External object files for target demo_ByteTrack
demo_ByteTrack_EXTERNAL_OBJECTS =

demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/bytetrack/src/BYTETracker.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/bytetrack/src/STrack.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/bytetrack/src/kalmanFilter.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/bytetrack/src/lapjv.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/bytetrack/src/utils.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/main.cpp.o
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/build.make
demo_ByteTrack: /usr/local/lib/libopencv_highgui.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_ml.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_objdetect.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_photo.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_stitching.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_video.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_videoio.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_imgcodecs.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_calib3d.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_dnn.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_features2d.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_flann.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_imgproc.so.4.8.1
demo_ByteTrack: /usr/local/lib/libopencv_core.so.4.8.1
demo_ByteTrack: CMakeFiles/demo_ByteTrack.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable demo_ByteTrack"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_ByteTrack.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo_ByteTrack.dir/build: demo_ByteTrack
.PHONY : CMakeFiles/demo_ByteTrack.dir/build

CMakeFiles/demo_ByteTrack.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo_ByteTrack.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo_ByteTrack.dir/clean

CMakeFiles/demo_ByteTrack.dir/depend:
	cd /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug /home/ai/data/zhangjunming/cpp_project/demo_ByteTrack/cmake-build-debug/CMakeFiles/demo_ByteTrack.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo_ByteTrack.dir/depend

