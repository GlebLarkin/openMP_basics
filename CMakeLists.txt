cmake_minimum_required(VERSION 3.20)
project(SYSTEMS-OF-LINEAL_EQUATIONS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)