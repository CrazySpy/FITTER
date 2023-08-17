cmake_minimum_required(VERSION 3.17)
project(EmbeddingBased)

set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    add_compile_options(/bigobj)
    add_compile_options(/arch:AVX)
elseif(MINGW)
    add_compile_options(-Wa,-mbig-obj)
    add_compile_options(-march=native)
endif ()

add_executable(EmbeddingBased main.cpp EmbeddingBased.cpp EmbeddingBased.h Types.h Utils.h Utils.cpp Simulator.cpp Simulator.h FCM.h FCM.cpp EventClock/EventClock.hpp)

find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(NOT APPLE)
    set(MKL_INTERFACE lp64)
    set(MKL_LINK static)
    find_package(MKL REQUIRED)
    add_definitions(-DEIGEN_USE_MKL_ALL)
    MESSAGE("hahah")
endif()

target_link_libraries(EmbeddingBased Eigen3::Eigen)
if(NOT APPLE)
    target_link_libraries(EmbeddingBased ${mkl_core_file} ${mkl_intel_lp64_file} ${mkl_intel_thread_file} ${MKL_THREAD_LIB})
    target_include_directories(EmbeddingBased PUBLIC ${MKL_INCLUDE})
endif()
