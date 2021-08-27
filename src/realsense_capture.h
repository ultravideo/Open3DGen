#pragma once

#include <iostream>
#include <Open3D/Open3D.h>
#include <librealsense2/rs.hpp>

#include "constants.h"

using namespace stitcher3d;

struct rgbd_data
{
    uint rgb_width;
    uint rgb_height;
    uint depth_width;
    uint depth_height;
    uint fps;

    /**
     *  TODO: post processing flags, laser power, etc
     */
};

class RGBDCamera
{
public:
    RGBDCamera(const rgbd_data& fdata);
    ~RGBDCamera();

    /**
     *  @note <rgb, depth>
     */
    std::pair<cv::Mat, cv::Mat> get_rgbd_aligned_cv() const;

    o3d::camera::PinholeCameraIntrinsic get_intrinsics() const;

    void adjust_to_auto_exposure() const;

private:
    rgbd_data m_camera_data;

    rs2::config m_config;
    rs2::pipeline m_pipeline;
    rs2::pipeline_profile m_profile;

    double m_depth_scale;

    rs2::align m_align;

};