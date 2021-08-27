cd build
gdb -ex run --args ./Open3DGen in ../data_files/capture/demos/tc_couch/ intr ../data_files/realsense_intrinsic_1280.matrix out ../data_files/out/ refine_cameras false skip_stride 15 unwrap true pipeline full poisson_depth 6 out_res 8192 depth_far_clip 3.0 cchain_len 3 lam_min_count 20
cd ..