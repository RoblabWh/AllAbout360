cmake_minimum_required(VERSION 3.9)
project(dfe2eqr_mapping LANGUAGES CXX)

find_package(OpenCV 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CPU_EXE dfe2eqr)
set(CUDA_EXE dfe2eqr-cuda)

add_executable(${CPU_EXE} src/${CPU_EXE}.cpp)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${CPU_EXE} PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
else()
	target_link_libraries(${CPU_EXE} ${OpenCV_LIBS})
endif()

enable_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	add_executable(${CUDA_EXE} src/${CUDA_EXE}.cpp src/mapping.cu src/mapping.h)
	target_link_libraries(${CUDA_EXE} ${OpenCV_LIBS})
endif()

