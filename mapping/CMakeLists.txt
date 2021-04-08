cmake_minimum_required(VERSION 3.9)
project(dfe2eqr_mapping LANGUAGES CXX)

include(CheckLanguage)

# find needed and optional packages
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
	find_package(OpenCV 3 REQUIRED)
endif()
find_package(OpenMP)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# set compile options
set(CMAKE_CXX_FLAGS "-std=gnu++17")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#----------------------------------------------------------------------------------#
# Functions to compile with different frameworks
#----------------------------------------------------------------------------------#
function(compile_with_openmp target_name)
	add_executable(${target_name} src/dfe2eqr.cpp)
	target_link_libraries(${target_name} ${OpenCV_LIBS})
endfunction()

function(compile_with_cuda target_name)
	add_executable(${target_name} src/dfe2eqr.cpp src/mapping.h src/mapping.cu)
	target_link_libraries(${target_name} ${OpenCV_LIBS})
endfunction()

function(compile_with_opencl target_name)
	message("OpenCL is currently not supported.")
	return()
endfunction()

#----------------------------------------------------------------------------------#
# Configure different compilation settings
#----------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(PRINT_TIMES "Print frametimes" NO)
if (${PRINT_TIMES})
	add_compile_options(-DPRINT_TIMES)
endif()

set(PARALLELIZATION_FRAMEWORK DEFAULT CACHE STRING "Parallelization framework to compile with.")
string(TOUPPER ${PARALLELIZATION_FRAMEWORK} PARALLELIZATION_FRAMEWORK)

# auto select best framework
if(${PARALLELIZATION_FRAMEWORK} STREQUAL "DEFAULT")
	if(CMAKE_CUDA_COMPILER)
		set(PARALLELIZATION_FRAMEWORK "CUDA")
	else()
		set(PARALLELIZATION_FRAMEWORK "OMP")
	endif()
endif()

# print configuration
message("\nCMake build type: " ${CMAKE_BUILD_TYPE}
		"\nParallelization framework: " ${PARALLELIZATION_FRAMEWORK}
		"\nPrint times: " ${PRINT_TIMES} "\n")

# create selected targets
if(${PARALLELIZATION_FRAMEWORK} STREQUAL "ALL")
	compile_with_openmp(${PROJECT_NAME}_omp)
	compile_with_cuda(${PROJECT_NAME}_cuda)
	# compile_with_opencl(${PROJECT_NAME}_cl)
elseif(${PARALLELIZATION_FRAMEWORK} STREQUAL "OPENMP" OR ${PARALLELIZATION_FRAMEWORK} STREQUAL "OMP")
	compile_with_openmp(${PROJECT_NAME})
elseif(${PARALLELIZATION_FRAMEWORK} STREQUAL "CUDA")
	compile_with_cuda(${PROJECT_NAME})
elseif(${PARALLELIZATION_FRAMEWORK} STREQUAL "OPENCL" OR ${PARALLELIZATION_FRAMEWORK} STREQUAL "CL")
	compile_with_opencl(${PROJECT_NAME})
else()
	message(${PARALLELIZATION_FRAMEWORK} " is not supported.")
endif()
