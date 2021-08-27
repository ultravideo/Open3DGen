cd build
cmake --build . --parallel 12 --config Debug
./Open3DGen in ../data_files/capture/demo_rock/ intr ../data_files/realsense_intrinsic_1280.matrix out ../data_files/out/ refine_cameras false reject_blur 1.1 crop false skip_stride 28 unwrap true pipeline full poisson_depth 7 simplify_voxel_size 0.075 out_res 4096 depth_far_clip 3.0 max_feature_count 2000
