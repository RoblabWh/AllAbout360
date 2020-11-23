cmake_minimum_required(VERSION 3.9)
project(dfe2eqr_calibration LANGUAGES CXX)

find_package(OpenCV 3 REQUIRED)
find_package(OpenMP)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
 
set(EXE dfe2eqr-calib)

add_executable(${EXE} ${EXE}.cpp)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${EXE} PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
else()
	target_link_libraries(${EXE} ${OpenCV_LIBS})
endif()