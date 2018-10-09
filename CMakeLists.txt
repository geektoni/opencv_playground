cmake_minimum_required(VERSION 3.5)
project(opencv)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(first_exercise first_exercise.cpp)
add_executable(background_sub bg_sub/main.cpp bg_sub/bg.cpp)
add_executable(background_sub_gauss bg_sub_gaussians/main.cpp bg_sub_gaussians/bg.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(first_exercise ${OpenCV_LIBS})
target_link_libraries(background_sub ${OpenCV_LIBS})
target_link_libraries(background_sub_gauss ${OpenCV_LIBS})