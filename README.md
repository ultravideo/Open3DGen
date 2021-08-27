# Open3DGen

## Building
Ubuntu 21.04 -based Linux distros should work. Tested on Pop-os 21.04. 

### Hardware Requirements
OpenGL 4.6 capable GPU, any decently modern GPU should do. With higher resolution textures (4k, 8k) 64Gb of RAM is recommended. This requiremenet will be fixed in a future release. Multicore CPU is recommended, as Open3DGen is highly parallelized.

</br>

Recommended RAM amounts: 
- 2k textures: 16Gb
- 4k textures: 32Gb
- 8k textures: 64Gb

### Dependencies
The following dependencies can be found in the repositories:
```
libboost1.71-all-dev
libopencv-dev
libopencv-contrib-dev
libopen3d-dev
libglfw-dev
libglew-dev
libopengl-dev
```
and can be installed with the command
```
sudo apt install -y libboost1.71-all-dev && sudo apt install -y libopencv-dev && sudo apt install -y libopencv-contrib-dev && sudo apt install -y libopen3d-dev && sudo apt install -y libglfw3-dev && sudo apt install -y libglew-dev && sudo apt install -y libopengl-dev
```
Installing GTSAM requires building from source: https://github.com/borglab/gtsam.

Make sure clang is also installed. Clang12 was used on the development platform. 


### Build
At this point building the project should be as simple as running `./build.sh`.

### Notes for Developers

If LanguageServerProtocol is used for code analysis and autocompletion, compile_commands can be generated with

```
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cd ..
ln build/compile_commands.json .
```
</br>

A quick way to compile or compile and run is to use the shell scripts `./only_compile.sh` and `./compile_and_run.sh`. Make sure to modify `compile_and_run.sh` to use the correct command line arguments.

## How to Use
Open3DGen requires at least 3 command line arguments to run:
```
in 'path to input rgbd sequence'
intr 'path to camera intrinsics'
out 'path to the output directory'
```

For example, </br>

`./build/Open3DGen in data_files/sequence intr data_files/realsense_intrinsic_1280.matrix out out/ ` </br>

The input RGBD sequence is expected to be separated into 2 folders: rgb/ and depth/. The RGB images should be named in the format of `rgb_number.png` (e.g. `rgb_0.png`) and the depth images should be in the format of `depth_number.png` (e.g. `depth_0.png`). The corresponding rgb and depth image must have the same number, and the numbers are expected to be in ascending order. 
</br>

The `camera_intrinsics.matrix` -file contains the image dimensions and the calibrated camera intrinsics. The expected format is 
```
width;height
fx;0.0;cx
0.0;fy;cy;
0.0;0.0;1.0
```

</br>

For example, `realsense_intrinsic_1280.matrix` contains 
```
1280;720
912.91984103;0.0;637.74400638
0.0;912.39513184;358.96757428
0.0;0.0;1.0
```

</br>

Do not modify the values set in the texture projection shader, correct values are loaded automatically.

The visualization can be disabled by commenting the line `#define DEBUG_VISUALIZE` in the file `src/constants.h`.

## Demo Scene
A demo RGB-D sequence is provided [here](http://ultravideo.fi/open3dgen/demo_rock.tar.xz). Extract the `demo_rock` folder into `data_files/`, and execute the following command.

`./build/Open3DGen in data_files/demo_rock/ intr data_files/realsense_intrinsic_1280.matrix out data_files/out/ refine_cameras false reject_blur 1.1 crop false skip_stride 28 unwrap true pipeline full poisson_depth 7 simplify_voxel_size 0.075 out_res 4096 depth_far_clip 3.0 max_feature_count 2000 assets_path data_files/`

The resulting `.obj`, `.mtl` and textures will be in the folder `data_files/out/`. It is recommended the `_dilated.png` texture is used. Use e.g. Blender to visualize the results.

The RGB-D sequence is not particularly high quality and contains an unfortunate amount of blur. Better sequence will be provided when the weather so allows. 
The result should look like this
![demo_rock_out](/images/demo_rock_out.png)

</br>

### Command Line Arguments
The supported command line arguments are as follows

- image sequence input path (`in`)
- camera input intrinsics (`intr`)
- output path (`out`)
- output texture resolution (`out_res`, currently supports only 2048, 4096, and 8192)
- use only every nth camera view to project textures (`project_every_nth`, 1 by default (= project from every view))
- mesh poisson depth reconstruction depth (`poisson_depth`, 9 by default)
- specify the mesh simplification (`simplify_voxel_size`, disabled by default (`0.0`), in meters, e.g. 5cm would be 0.05. Larger values result in lower quality but speed up the texture projection significantly. 5cm-15cm is generally a good range of values to try)
- export intermediary mesh, before texture projection (`export_mesh_interm`, `true`/`false`, default `false`)
- use a specific mesh for projection (`mesh_path`)
- the parts of the pipeline that will be run (`pipeline`, options are `full`, `only_pointcloud`, `only_mesh`, or `only_project`)
	- `full` runs the entire sequence and the output is a textured mesh
	- `only_pointcloud` localizes the cameras and creates a pointcloud. Camera positions are serialized into a file in the `out` -folder
	- `only_mesh` requires the cameras to be already serialized. 
	- `only_project` only projects the textures, useful for editing the mesh before projection, e.g. re-topology or manually creating uv -coordinates
- specify camera distortion coefficients (`dist_coeff 'path to file'`, uses zero distortion by default)
	- loads distortion coefficients from a file, coefficients should be space separated e.g. `0.66386586e-01, -5.26413220e-01, -1.01376611e-03, 1.59777094e-04, 4.65208008e-01`
- feature candidate chain length (`cchain_len`, 7 by default). Specifies through how many frames should a feature be tracked before it can be made into a landmark.
- feature landmark match min (`lam_min_count`, 20 by default). Specifies how many features should be matched to landmarks for a frame to be considered succesful.
- serialized camera and pointcloud (`serialized_cameras_path`, where the serialized cameras and pointcloud should be saved/loaded from)
- assets path (`assets_path`, where to load assets from, data_files/ -by default)
- load mesh from path (`load_mesh_path`, none by default. Loads a mesh from given path for projection)
- unwrap UVs (`unwrap`, options `true`/`false`, `true` by default)
- refine camera pose after adding cameras (`refine_cameras`, `true`/`false`, `false` by default)
- texture projection shader work group size (`workgroup_size`, default is 1024)
- skip first `n` frames after the frist frame (`skip_stride`), after the first frame skips `n` frames, used to help force camera pose quality
- write intermediary per-frame projected textures (`write_interm_images`, `true`/`false`, `false` by default)
- set the depth far clipping distance (`depth_far_clip`, units in meters, default is `3.6`)
- set the maximum number of features detected (`max_feature_count`, default is unlimited (`0`), which is overkill. Note: this can be slow especially in outdoor scenes. To significantly increase performance, limit this one to a more reasonable value, e.g. `1000`)
- remove the poisson reconstruction "ballooning" artefacts (`crop`, `true`/`false`, default is `true`. May not always work correctly and removes too much detail.)
- texture projection ray threshold (`ray_threshold`, default is `0.9`, which will project everything except nearly 90d angles, the dot-product between face normal and the projection ray, after which the result is discarded. Larger values project more, but result in more smearing in the textures)
- texture projection blur threshold (`reject_blur`, default is disabled `0.0`. A value of `1.0` means frames with sharpness less than the average sharpness will be rejected, values greater than `1.0` will thus result in more frames getting rejected)


### Known issues and TODO
- The real-time camera pose estimation gets progressively slower as more frames are added due to the increased amount of landmarks
	- implement a more efficient landmark search algorithm
- Add support for the realsense API and real-time capture and reconstruction
- SIFT is not particularly fast
	- experiment with SURF and/or ORB
- The texture projection stage runs out of system RAM
	- load and save the projected textures individually instead of acting on the entire sequence at once
- Xatlas produces poor UVs
	- if we cannot get Xatlas to keep connected triangles connected in the UV -map as well, the plan is to phase out Xatlas at some point in favor of some other solution
- Also planned in the next release is a better texture projection algorithm based on inverse-matrices. This should not only be faster, but more scalable for very large and high-poly meshes as well. Additionally, this would reduce the issue of UV bleeding.
- Implement a way of smoothing noisy depth maps, decrease the effect of "edge bleed" on sharp depth transitions. Could view synthesis be used for this? 


## Paper
If you are using Open3DGen in your research, please refer to the following paper: <br/>
`T. T. Niemirepo, M. Viitanen, and J. Vanne, “Open3DGen: Open-Source software for reconstructing textured 3D models from RGB-D images,” Accepted to ACM Multimedia Syst. Conf., Istanbul, Turkey, Sept.-Oct. 2021. `
