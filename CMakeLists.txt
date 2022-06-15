cmake_minimum_required(VERSION 3.17)
project(EmbeddingBased)

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    add_compile_options(/bigobj)
else ()
    add_compile_options(-Wa,-mbig-obj)
endif ()

find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(EmbeddingBased main.cpp EmbeddingBased.cpp EmbeddingBased.h Types.h Utils.h Utils.cpp Simulator.cpp Simulator.h FCM.h FCM.cpp)
target_link_libraries(EmbeddingBased Eigen3::Eigen)
target_link_libraries(EmbeddingBased ${OpenCV_LIBS})