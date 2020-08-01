cmake_minimum_required(VERSION 3.5)

project(nonlinear_equation ${PROJECT_NAME} LANGUAGES CXX)

set(SOURCES
    main.cpp)
set(HEADERS
    include/base_solver.hpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=14800")

if(NOT DEFINED OpenGL_GL_PREFERENCE)
  set(OpenGL_GL_PREFERENCE "LEGACY")
endif()

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS context coroutine system regex)


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
