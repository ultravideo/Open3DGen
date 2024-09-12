# EXPERIMENTAL Open3DGen Dockerfile to compile and run the code
# docker build . -t open3dgen
# docker run --gpus all -v /local_datafolder/:/data_files --rm -it open3dgen
# Use with the demo_rock example data:
# /app/build/Open3DGen in /data_files/demo_rock/ intr /data_files/realsense_intrinsic_1280.matrix out /data_files/out/ refine_cameras false reject_blur 1.1 crop false skip_stride 28 unwrap true pipeline full poisson_depth 7 simplify_voxel_size 0.075 out_res 4096 depth_far_clip 3.0 max_feature_count 750 assets_path /data_files/
FROM ubuntu:24.04

WORKDIR /app

RUN apt update && apt upgrade -y

RUN apt install -y libboost1.83-all-dev && \
    apt install -y libopencv-dev && \
    apt install -y libopencv-contrib-dev && \
    apt install -y libglfw3-dev && \
    apt install -y libglew-dev && \
    apt install -y libopengl-dev && \
    apt install -y libeigen3-dev libgtsam-dev && \
    apt install -y libc++-dev && \
    apt install -y git build-essential cmake wget

RUN git clone https://github.com/borglab/gtsam.git && cd gtsam && \
    mkdir build && cd build && cmake .. && make install && \
    cd ../.. && rm -rf gtsam


RUN wget https://github.com/isl-org/Open3D/releases/download/v0.18.0/open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.xz && \
    tar -xvf open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.xz && \
    rm open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.xz && \
    cp -rf open3d-devel-linux-x86_64-cxx11-abi-0.18.0/* /usr/ && \
    rm -rf open3d-devel-linux-x86_64-cxx11-abi-0.18.0

# To build open3d from the source
#RUN wget https://github.com/isl-org/Open3D/archive/refs/tags/v0.18.0.tar.gz && \
#    tar -xvf v0.18.0.tar.gz && \
#    mkdir Open3D-0.18.0/build && cd Open3D-0.18.0/build && \
#    cmake -DENABLE_HEADLESS_RENDERING=ON \
#      -DBUILD_GUI=OFF \
#      -DUSE_SYSTEM_GLEW=OFF \
#      -DUSE_SYSTEM_GLFW=OFF \
#      -DBUILD_EXAMPLES=OFF \
#      -DBUILD_PYTHON_MODULE=OFF  -DBUNDLE_OPEN3D_ML=OFF   -DBUILD_PYTORCH_OPS=OFF \
#      -DCMAKE_CXX_FLAGS=-fpermissive -Wno-free-nonheap-object \
#      .. && \
#    make -j8 && make install && \
#    cd ../.. && rm -rf Open3D-0.18.0 v0.18.0.tar.gz
#

# Add the bare minimum to compile the code
ADD CMakeLists.txt /app
ADD main.cpp /app/main.cpp
ADD src /app/src
ADD data_files /app/data_files
ADD libraries /app/libraries

RUN cd /app && mkdir build && cd build && cmake -DHEADLESS=ON .. && cmake --build . --parallel 12 --config Release

ENTRYPOINT ["/bin/bash"]
