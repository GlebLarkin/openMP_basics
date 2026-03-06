add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(${PROJECT_NAME} INTERFACE
        -O3
        -march=native
)

target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)