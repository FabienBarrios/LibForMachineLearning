# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


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

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\Application\JetBrains\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = D:\Application\JetBrains\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Projet\V2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Projet\V2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\V2.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\V2.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\V2.dir\flags.make

CMakeFiles\V2.dir\library.cpp.obj: CMakeFiles\V2.dir\flags.make
CMakeFiles\V2.dir\library.cpp.obj: ..\library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Projet\V2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/V2.dir/library.cpp.obj"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\V2.dir\library.cpp.obj /FdCMakeFiles\V2.dir\ /FS -c D:\Projet\V2\library.cpp
<<

CMakeFiles\V2.dir\library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/V2.dir/library.cpp.i"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\V2.dir\library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Projet\V2\library.cpp
<<

CMakeFiles\V2.dir\library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/V2.dir/library.cpp.s"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\V2.dir\library.cpp.s /c D:\Projet\V2\library.cpp
<<

CMakeFiles\V2.dir\CalculMatriciel.cpp.obj: CMakeFiles\V2.dir\flags.make
CMakeFiles\V2.dir\CalculMatriciel.cpp.obj: ..\CalculMatriciel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Projet\V2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/V2.dir/CalculMatriciel.cpp.obj"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\V2.dir\CalculMatriciel.cpp.obj /FdCMakeFiles\V2.dir\ /FS -c D:\Projet\V2\CalculMatriciel.cpp
<<

CMakeFiles\V2.dir\CalculMatriciel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/V2.dir/CalculMatriciel.cpp.i"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\V2.dir\CalculMatriciel.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Projet\V2\CalculMatriciel.cpp
<<

CMakeFiles\V2.dir\CalculMatriciel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/V2.dir/CalculMatriciel.cpp.s"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\V2.dir\CalculMatriciel.cpp.s /c D:\Projet\V2\CalculMatriciel.cpp
<<

CMakeFiles\V2.dir\PMC.cpp.obj: CMakeFiles\V2.dir\flags.make
CMakeFiles\V2.dir\PMC.cpp.obj: ..\PMC.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Projet\V2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/V2.dir/PMC.cpp.obj"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\V2.dir\PMC.cpp.obj /FdCMakeFiles\V2.dir\ /FS -c D:\Projet\V2\PMC.cpp
<<

CMakeFiles\V2.dir\PMC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/V2.dir/PMC.cpp.i"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\V2.dir\PMC.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Projet\V2\PMC.cpp
<<

CMakeFiles\V2.dir\PMC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/V2.dir/PMC.cpp.s"
	C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\V2.dir\PMC.cpp.s /c D:\Projet\V2\PMC.cpp
<<

# Object files for target V2
V2_OBJECTS = \
"CMakeFiles\V2.dir\library.cpp.obj" \
"CMakeFiles\V2.dir\CalculMatriciel.cpp.obj" \
"CMakeFiles\V2.dir\PMC.cpp.obj"

# External object files for target V2
V2_EXTERNAL_OBJECTS =

V2.dll: CMakeFiles\V2.dir\library.cpp.obj
V2.dll: CMakeFiles\V2.dir\CalculMatriciel.cpp.obj
V2.dll: CMakeFiles\V2.dir\PMC.cpp.obj
V2.dll: CMakeFiles\V2.dir\build.make
V2.dll: CMakeFiles\V2.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Projet\V2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library V2.dll"
	D:\Application\JetBrains\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe -E vs_link_dll --intdir=CMakeFiles\V2.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\mt.exe --manifests -- C:\PROGRA~2\MICROS~3\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\link.exe /nologo @CMakeFiles\V2.dir\objects1.rsp @<<
 /out:V2.dll /implib:V2.lib /pdb:D:\Projet\V2\cmake-build-debug\V2.pdb /dll /version:0.0 /machine:x64 /debug /INCREMENTAL  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  
<<

# Rule to build all files generated by this target.
CMakeFiles\V2.dir\build: V2.dll

.PHONY : CMakeFiles\V2.dir\build

CMakeFiles\V2.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\V2.dir\cmake_clean.cmake
.PHONY : CMakeFiles\V2.dir\clean

CMakeFiles\V2.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" D:\Projet\V2 D:\Projet\V2 D:\Projet\V2\cmake-build-debug D:\Projet\V2\cmake-build-debug D:\Projet\V2\cmake-build-debug\CMakeFiles\V2.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\V2.dir\depend

