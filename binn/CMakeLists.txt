cmake_minimum_required(VERSION 3.22)

project(new VERSION 0.1.0)

add_executable(binn binn.cpp)
find_package(OpenMP REQUIRED)
find_package(fmt CONFIG REQUIRED)

include_directories(../binn)
add_library(interface INTERFACE)

target_compile_features(interface INTERFACE cxx_std_23)
set_target_properties(interface PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(interface INTERFACE OpenMP::OpenMP_CXX fmt::fmt)

target_link_libraries(binn PRIVATE interface)
