cmake_minimum_required(VERSION 3.9)
project(dfe LANGUAGES CXX)

include(CheckLanguage)

# find needed and optional packages
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
	find_package(OpenCV 3 REQUIRED)
endif()
find_package(OpenMP)
find_package(OpenCL)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# set compile options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#----------------------------------------------------------------------------------#
# Functions to compile with different frameworks
#----------------------------------------------------------------------------------#
function(mapping_serial target_name)
	add_executable(${target_name}-mapper src/mapping.cpp)
	add_library(${target_name}-mapping SHARED src/mapping.cpp)
	target_link_libraries(${target_name}-mapper ${OpenCV_LIBS} -lpthread)
	target_link_libraries(${target_name}-mapping ${OpenCV_LIBS})
endfunction()

function(mapping_with_openmp target_name)
	if(OpenMP_CXX_FOUND)
		add_executable(${target_name}-mapper src/mapping.cpp)
		add_library(${target_name}-mapping SHARED src/mapping.cpp)
		target_link_libraries(${target_name}-mapper PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} -lpthread)
		target_link_libraries(${target_name}-mapping PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
	else()
		message("OpenMP not available")
	endif()
endfunction()

function(mapping_with_cuda target_name)
	if(CMAKE_CUDA_COMPILER)
		add_executable(${target_name}-mapper src/mapping.cpp src/mapping.cu)
		if (OpenMP_CXX_FOUND)
			target_link_libraries(${target_name}-mapper PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} -lpthread)
		else()
			target_link_libraries(${target_name}-mapper PUBLIC ${OpenCV_LIBS} -lpthread)
		endif()
		target_compile_definitions(${target_name}-mapper PUBLIC ON_GPU WITH_CUDA)

		message("Library support is currently not available for CUDA.")
		# add_library(${target_name}-mapping SHARED src/mapping.cpp src/mapping.cu)
		# target_link_libraries(${target_name}-mapping PUBLIC ${OpenCV_LIBS})
		# target_compile_definitions(${target_name}-mapping PUBLIC ON_GPU WITH_CUDA)
	else()
		message("CUDA not available")
	endif()
endfunction()

function(mapping_with_opencl target_name)
	if(OpenCL_FOUND)
		include_directories(${OpenCL_INCLUDE_DIRS} build)
		add_custom_command(OUTPUT build/mapping.cl.h
						COMMAND printf \"\#ifndef MAPPING_CL_H\\n\#define MAPPING_CL_H\\n\" > build/mapping.cl.h
						COMMAND xxd -i src/mapping.cl >> build/mapping.cl.h
						COMMAND printf \"\#endif\\n\" >> build/mapping.cl.h
						COMMAND sed -i 's\\unsigned\\static unsigned\\g' build/mapping.cl.h
						DEPENDS src/mapping.cl
						WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
		add_custom_target(generate-mapping.cl.h DEPENDS build/mapping.cl.h)

		add_executable(${target_name}-mapper src/mapping.cpp)
		if (OpenMP_CXX_FOUND)
			target_link_libraries(${target_name}-mapper PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${OpenCL_LIBRARIES} -lpthread)
		else()
			target_link_libraries(${target_name}-mapper PUBLIC ${OpenCV_LIBS} ${OpenCL_LIBRARIES} -lpthread)
		endif()
		target_compile_definitions(${target_name}-mapper PUBLIC ON_GPU WITH_OPENCL)
		add_dependencies(${target_name}-mapper generate-mapping.cl.h)

		add_library(${target_name}-mapping SHARED src/mapping.cpp)
		if (OpenMP_CXX_FOUND)
			target_link_libraries(${target_name}-mapping OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
		else()
			target_link_libraries(${target_name}-mapping ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
		endif()
		target_compile_definitions(${target_name}-mapping PUBLIC ON_GPU WITH_OPENCL)
		add_dependencies(${target_name}-mapping generate-mapping.cl.h)
	else()
		message("OpenCL not available")
	endif()
endfunction()

#----------------------------------------------------------------------------------#
# Configure different compilation settings
#----------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(PRINT_TIMES "Print frametimes" NO)
string(TOUPPER ${PRINT_TIMES} PRINT_TIMES)
if (${PRINT_TIMES})
	add_compile_options(-DPRINT_TIMES)
endif()

set(PARALLELIZATION DEFAULT CACHE STRING "Parallelization framework to compile with.")
string(TOUPPER ${PARALLELIZATION} PARALLELIZATION)

# auto select best framework
if(${PARALLELIZATION} STREQUAL "DEFAULT")
	if(OpenCL_FOUND)
		set(PARALLELIZATION "CL")
	# elseif(CMAKE_CUDA_COMPILER)
	# 	set(PARALLELIZATION "CUDA")
	elseif (OpenMP_CXX_FOUND)
		set(PARALLELIZATION "OMP")
	else()
		set(PARALLELIZATION "OFF")
	endif()
endif()

# print configuration
message("<------ Configuration ------>"
		"\nCMake Build Type: " ${CMAKE_BUILD_TYPE}
		"\nParallelization: " ${PARALLELIZATION}
		"\nPrint Times: " ${PRINT_TIMES}
		"\n<------- Information ------->")

# create selected mapping target
if(${PARALLELIZATION} STREQUAL "ALL")
	mapping_serial(${PROJECT_NAME})
	mapping_with_openmp(${PROJECT_NAME}-omp)
	mapping_with_cuda(${PROJECT_NAME}-cuda)
	mapping_with_opencl(${PROJECT_NAME}-cl)
elseif(${PARALLELIZATION} STREQUAL "OFF")
	mapping_serial(${PROJECT_NAME})
elseif(${PARALLELIZATION} STREQUAL "OPENMP" OR ${PARALLELIZATION} STREQUAL "OMP")
	mapping_with_openmp(${PROJECT_NAME})
elseif(${PARALLELIZATION} STREQUAL "CUDA")
	mapping_with_cuda(${PROJECT_NAME})
elseif(${PARALLELIZATION} STREQUAL "OPENCL" OR ${PARALLELIZATION} STREQUAL "CL")
	mapping_with_opencl(${PROJECT_NAME})
else()
	message(${PARALLELIZATION} " is not supported.")
endif()

message("<--------------------------->")

# create calibration target
add_executable(${PROJECT_NAME}-calibration src/calibration.cpp)
if (OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_NAME}-calibration OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${PROJECT_NAME}-mapping)
else()
	target_link_libraries(${PROJECT_NAME}-calibration ${OpenCV_LIBS} ${PROJECT_NAME}-mapping)
endif()

# create example usage
if (NOT ${PARALLELIZATION} STREQUAL "CUDA")
	add_executable(example src/example.cpp)
	target_link_libraries(example ${PROJECT_NAME}-mapping)
endif()