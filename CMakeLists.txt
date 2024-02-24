cmake_minimum_required(VERSION 3.8)
project(the_sdoajalizer LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 20)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
# find_package(CUDA 10 REQUIRED)
# include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

# Set up include and lib paths
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
########################################

find_package(OpenGL REQUIRED)

set(EXTERNAL "external")

set(GLFW_ROOT_DIR ${EXTERNAL})
set(GLFW_USE_STATIC_LIBS ON)
find_package(GLFW REQUIRED)

set(GLEW_ROOT_DIR ${EXTERNAL})
set(GLEW_USE_STATIC_LIBS ON)
find_package(GLEW REQUIRED)

add_definitions(${GLEW_DEFINITIONS})
include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

########################################

file(GLOB_RECURSE headers "src/*.h" "src/*.hpp")
file(GLOB_RECURSE sources "src/*.cpp" "src/*.cu")

list(SORT headers)
list(SORT sources)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Headers" FILES ${headers})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Sources" FILES ${sources})

include_directories("${CMAKE_SOURCE_DIR}/src")

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBRARIES} nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")

########################################

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/fonts ${CMAKE_BINARY_DIR}/fonts)
