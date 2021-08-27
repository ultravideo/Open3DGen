#pragma once

#include <Open3D/Open3D.h>
// #include <open3d/pipelines/registration/ColoredICP.h>
#include <Open3D/Registration/ColoredICP.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <functional>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include "constants.h"
#include "surface_mesh.h"
#include "utilities.h"
#include "rgbd_utilities.h"
#include "timer.h"
#include "math_modules.h"
#include "visualizer.h"

#include "poseg2.h"
#include "realtime_mesh.h"
#include "registration_params.h"

namespace stitcher3d
{
namespace registration
{

Eigen::Matrix4d_u colored_ICP(const Eigen::Matrix4d& initial_transform, const o3d::geometry::PointCloud& source,
    const o3d::geometry::PointCloud& target, const float voxel_size);

Eigen::Matrix4d_u registration_ICP(const Eigen::Matrix4d& initial_transform, const o3d::geometry::PointCloud& source,
    const o3d::geometry::PointCloud& target, const float voxel_size);

std::tuple<std::shared_ptr<surface::SurfaceMesh>, std::vector<std::shared_ptr<Camera>>, float>
    register_pointcloud_sequence_features(const std::string& rgbd_path, const o3d::camera::PinholeCameraIntrinsic& intr,
    const std::vector<double>& distortion_coeffs, const RegistrationParams& rparams);

std::tuple<Eigen::Matrix4d_u, bool> register_pointclouds_slow(const o3d::geometry::PointCloud& source, 
                                    const o3d::geometry::PointCloud& target,
                                    const float voxel_size, const float fitness_err_thr=0.55f,
                                    const int max_iter_count=15);

down_fpfh downsample_and_fpfh_pcloud(const o3d::geometry::PointCloud& cloud, const float voxel_size);


std::shared_ptr<surface::SurfaceMesh> register_rgbd_sequence(const std::string& rgbd_path,
                        const o3d::camera::PinholeCameraIntrinsic& intr, int frame_skip=1);


std::tuple<std::shared_ptr<surface::SurfaceMesh>,
    std::vector<o3d::geometry::Image>,
    std::vector<o3d::geometry::Image>,
    std::vector<Eigen::Matrix4d_u>> register_rgbd_sequence_for_surface(const std::string& rgbd_path,
                        const o3d::camera::PinholeCameraIntrinsic& intr, int frame_skip=1, int sequence_length=-1);

}
}