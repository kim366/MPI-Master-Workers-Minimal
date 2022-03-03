cmake_minimum_required(VERSION 3.8)
project(hello)

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

add_executable(hello src/main.cpp)

target_link_libraries(hello PRIVATE ${MPI_C_LIBRARIES})
target_include_directories(hello PRIVATE ${MPI_C_INCLUDE_PATH})

target_compile_options(hello PUBLIC -Wall -Wextra -Wconversion -pedantic)
set_target_properties(hello PROPERTIES CXX_STANDARD 17)
