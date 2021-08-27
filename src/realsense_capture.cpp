#include "realsense_capture.h"
#include "timer.h"

RGBDCamera::RGBDCamera(const rgbd_data& fdata)  :
    m_camera_data(fdata),
    m_config(rs2::config()),
    m_pipeline(rs2::pipeline()),
    m_align(rs2::align(RS2_STREAM_COLOR))
{
    m_config.enable_stream(RS2_STREAM_DEPTH, m_camera_data.depth_width, 
        m_camera_data.depth_height, RS2_FORMAT_Z16, m_camera_data.fps);
    m_config.enable_stream(RS2_STREAM_COLOR, m_camera_data.rgb_width, 
        m_camera_data.rgb_height, RS2_FORMAT_BGR8, m_camera_data.fps);

    m_pipeline.start(m_config);
    std::cout << "start pipeline\n";

    m_profile = m_pipeline.get_active_profile();
    std::cout << "get profile\n";

    auto depth_sensor = m_profile.get_device().query_sensors()[0].as<rs2::depth_sensor>();
    std::cout << "get depth sensor\n";

    depth_sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
    std::cout << "set high accuracy\n";
    m_depth_scale = depth_sensor.get_depth_scale();
    std::cout << "get depth scale\n";

    try
    {
        rs2::option_range laser_power_range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
        depth_sensor.set_option(RS2_OPTION_LASER_POWER, laser_power_range.max * 0.25f);
        std::cout << "set laser power\n";
    } catch (...) {  }

    m_align = rs2::align(RS2_STREAM_COLOR);
}

RGBDCamera::~RGBDCamera()
{
    auto depth_sensor = m_profile.get_device().query_sensors()[0].as<rs2::depth_sensor>();
    auto rgb_sensor = m_profile.get_device().query_sensors()[1].as<rs2::color_sensor>();
    
    depth_sensor.stop();
    rgb_sensor.stop();

    depth_sensor.close();
    rgb_sensor.close();    
}

o3d::camera::PinholeCameraIntrinsic RGBDCamera::get_intrinsics() const
{
    auto rgb_sensor = m_profile.get_device().query_sensors()[1].as<rs2::color_sensor>();
    rs2::video_stream_profile vsp = rgb_sensor.get_stream_profiles()[0].as<rs2::video_stream_profile>();
    rs2_intrinsics rs2_intr = vsp.get_intrinsics();

    const o3d::camera::PinholeCameraIntrinsic intr(rs2_intr.width, rs2_intr.height, 
        rs2_intr.fx, rs2_intr.fy, rs2_intr.ppx, rs2_intr.ppy);
    std::cout << intr.intrinsic_matrix_ << "\n";
    return intr;
}

std::pair<cv::Mat, cv::Mat> RGBDCamera::get_rgbd_aligned_cv() const
{
    rs2::frameset frames = m_pipeline.wait_for_frames();
    frames = m_align.process(frames);

    rs2::frame rgb = frames.get_color_frame();
    rs2::frame depth = frames.get_depth_frame();

    const uint8_t* rgb_data_begin = (uint8_t*)rgb.get_data();
    const uint16_t* depth_data_begin = (uint16_t*)depth.get_data();

    const cv::Mat cv_rgb (cv::Size(m_camera_data.rgb_width, m_camera_data.rgb_height), CV_8UC3, (void*)rgb_data_begin, cv::Mat::AUTO_STEP);
    const cv::Mat cv_depth (cv::Size(m_camera_data.rgb_width, m_camera_data.rgb_height), CV_16S, (void*)depth_data_begin, cv::Mat::AUTO_STEP);

    return std::make_pair(cv_rgb, cv_depth);
}

void RGBDCamera::adjust_to_auto_exposure() const
{
    for (int i = 0; i < 30; i++)
        const auto temp = get_rgbd_aligned_cv();
}