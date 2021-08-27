cmake_minimum_required(VERSION 3.10.2)
set (CMAKE_CXX_STANDARD 17)
# add_compile_options(-std=c++17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(SOURCES main.cpp src/constants.h src/rgbd_utilities.cpp src/rgbd_utilities.h src/image.h 
    src/surface_mesh.cpp src/surface_mesh.h src/pointcloud_registration.cpp src/pointcloud_registration.h 
    src/timer.h src/utilities.h src/utilities.cpp src/visualizer.cpp src/visualizer.h 
    src/uv_unwrapper.h src/uv_unwrapper.cpp src/math_modules.cpp src/math_modules.h src/texture_projector.cpp
    src/texture_projector.h src/shader_stuff.h src/shader_stuff.cpp 
    src/s3d_camera.h
    src/features.h src/features.cpp src/feature_graph.h
    src/realtime_mesh.h src/realtime_mesh.cpp
    src/poseg2.h src/poseg2.cpp
    src/registration_params.h
    src/pose_predictor.h src/pose_predictor.cpp
    libraries/xatlas/xatlas.h libraries/xatlas/xatlas.cpp
    libraries/tinyobj/tiny_obj_loader.h libraries/tinyobj/tiny_obj_loader.cpp
)

project(Open3DGen)

find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS} -fopenmp") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

add_definitions(-DUNIX)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-result)

# add_definitions(-g -fopenmp)
add_definitions(-O3 -fopenmp)

message(STATUS "Found Open3D ${Open3D_VERSION} ${Open3D_LIBRARY_DIRS}")


# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(GTSAM REQUIRED)

# this is not implemented yet
# find_package(realsense2 REQUIRED)

link_directories(${Open3D_LIBRARY_DIRS})
# include_directories(include ${realsense_INCLUDE_DIR})

include_directories(${GLEW_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR})

add_executable(Open3DGen ${SOURCES})

target_include_directories(Open3DGen PUBLIC ${Open3D_INCLUDE_DIRS} 
    # ${realsense_INCLUDE_DIR}
)

target_link_libraries(Open3DGen 
                        ${Open3D_LIBRARIES} 
                        ${OpenCV_LIBS}
                        gtsam
#                        ${realsense2_LIBRARY}
                        ${GLEW_LIBRARIES}
                        glfw
                        OpenGL::GL
                        Eigen3::Eigen
)
