cmake_minimum_required(VERSION 3.5)
project(opencv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(first_exercise first_exercise.cpp)
add_executable(background_sub bg_sub/main.cpp bg_sub/bg.cpp)
add_executable(background_sub_gauss bg_sub_gaussians/main.cpp bg_sub_gaussians/bg.cpp)
add_executable(mean_shift meanshift/meanShift_start.cpp)
add_executable(kalman kalman/KalmanFilter_start.cpp)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")

target_link_libraries(first_exercise opencv_core opencv_highgui opencv_video opencv_imgproc)
target_link_libraries(background_sub opencv_highgui opencv_core opencv_imgproc opencv_video)
target_link_libraries(background_sub_gauss opencv_highgui opencv_core opencv_imgproc opencv_video)
target_link_libraries(mean_shift opencv_highgui opencv_core opencv_imgproc opencv_video)
target_link_libraries(kalman opencv_highgui opencv_core opencv_imgproc opencv_video)