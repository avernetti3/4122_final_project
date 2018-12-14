cmake_minimum_required(VERSION 3.9.1)
project(p3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

find_package(MPI REQUIRED)
find_package(CUDA QUIET REQUIRED)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;)

include_directories(
	${MPI_INCLUDE_PATH}
	src
)

set(SOURCE_FILES_1
	src/p31.cpp
	src/input_image.h
	src/input_image.cc
	src/complex.h
	src/complex.cc
)

set(SOURCE_FILES_2
	src/p32.cpp
	src/input_image.h
	src/input_image.cc
	src/complex.h
	src/complex.cc
)

set(SOURCE_FILES_3
	src/p33.cu
	src/input_image.h
	src/input_image.cc
	src/complex.h
	src/complex.cc
)

add_executable(p31 ${SOURCE_FILES_1})
add_executable(p32 ${SOURCE_FILES_2})
CUDA_ADD_EXECUTABLE(p33 ${SOURCE_FILES_3})

target_link_libraries(p32 ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(p32 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(p32 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
