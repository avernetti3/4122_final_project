cmake_minimum_required(VERSION 3.9.1)
project(p32)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

#find_package(MPI REQUIRED)

include_directories(
	#${MPI_INCLUDE_PATH}
	src
)

set(SOURCE_FILES
	src/p32.cpp
	src/input_image.h
	src/input_image.cc
	src/complex.h
	src/complex.cc
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

#target_link_libraries(${PROJECT_NAME}${MPI_LIBRARIES})

#if(MPI_COMPILE_FLAGS)
#  set_target_properties(${PROJECT_NAME} PROPERTIES
#    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()

#if(MPI_LINK_FLAGS)
#  set_target_properties(${PROJECT_NAME} PROPERTIES
#    LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
