cmake_minimum_required (VERSION 3.0)
project (hubbard)
include_directories(src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wno-unused-variable -Wno-register -Werror")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED)

file(GLOB sources "src/*.cpp")

add_executable(hubbard ${sources})
target_include_directories(hubbard PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(hubbard PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries (hubbard Eigen3::Eigen)
target_link_libraries(hubbard ${PYTHON_LIBRARIES})
target_link_libraries(hubbard ${Boost_LIBRARIES})
