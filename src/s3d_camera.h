#pragma once
#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <Open3D/Geometry/Image.h>
#include <Open3D/Geometry/PointCloud.h>
#include <cstdint>

#include <Open3D/Open3D.h>
#include <memory>
// #include "../libraries/libmv/src/libmv/correspondence/feature_matching.h"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace o3d = open3d;

namespace stitcher3d
{

struct Camera
{
    virtual ~Camera() = default;

    /**
     * brief The absolute position of the camera in the world
     */
    Eigen::Vector3d position;
    /**
     * @brief The absolute rotation of the camera in the world
     */
    Eigen::Matrix3d rotation;

    /**
     * @brief The scale of of the camera translation
     */
    float scale;

    /**
     * @brief The relative translation from last frame
     */
    Eigen::Vector3d dt;

    /**
     * @brief The relative rotation from last frame
     */
    Eigen::Matrix3d dR;

    /**
     * @brief The absolute camera transform matrix in the world
     */
    Eigen::Matrix4d T;

    /**
     * @brief The camera projection matrix
     */
    Eigen::Matrix<double, 3, 4> P;

    std::vector<double> distortion_coeffs;

    std::shared_ptr<o3d::geometry::PointCloud> view_cloud;

    std::vector<cv::KeyPoint> kp_features;
    cv::Mat feature_descriptors;
    
    std::shared_ptr<o3d::geometry::Image> rgb;
    std::shared_ptr<o3d::geometry::Image> depth;

    std::shared_ptr<cv::Mat> cv_rgb;
    std::shared_ptr<cv::Mat> cv_depth;

    o3d::camera::PinholeCameraIntrinsic intr;

    int id;

    float rodriques_angle;

    std::string rgb_name;
    std::string depth_name;

    Camera(Eigen::Vector3d position, Eigen::Matrix3d rotation, 
        float scale, Eigen::Vector3d dt, Eigen::Matrix3d dR, 
        Eigen::Matrix4d T, Eigen::Matrix<double, 3, 4> P,
        std::vector<double> distortion_coeffs, std::shared_ptr<o3d::geometry::PointCloud> view_cloud,
        std::vector<cv::KeyPoint> kp_features, cv::Mat feature_descriptors,
        std::shared_ptr<o3d::geometry::Image> rgb, std::shared_ptr<o3d::geometry::Image> depth,
        std::shared_ptr<cv::Mat> cv_rgb, std::shared_ptr<cv::Mat> cv_depth, 
        o3d::camera::PinholeCameraIntrinsic intr, int id, float rodriques_angle, 
        std::string rgb_name, std::string depth_name) :
        
        position(position), rotation(rotation), scale(scale), dt(dt), dR(dR), T(T), P(P),
        distortion_coeffs(distortion_coeffs), view_cloud(view_cloud), kp_features(kp_features),
        rgb(rgb), depth(depth), cv_rgb(cv_rgb), cv_depth(cv_depth), intr(intr), id(id),
        rodriques_angle(rodriques_angle), rgb_name(rgb_name), depth_name(depth_name)
    {

    }

    Camera() { }
};

/**
 *  A camera model specific to the slow pose algorithm
 */
struct OfflineCamera : public Camera
{
    /**
     *  specifies, which keypoint index corresponds to which 3D landmark
     *  <keypoint_index, landmark_index>
     */
    std::map<uint32_t, uint32_t> landmark_lookup;

    bool landmark_kp_exists(const uint32_t kp_index)
    { return landmark_lookup.count(kp_index) != 0; }


    /**
     *  contains keypoint matches in reference to other frames,
     *  i.e. "where in other frames can this feature be found"
     * 
     *  <this_camera_kp_idx, <other_camera_idx, other_cam_kp_idx>>
     */
    std::map<uint32_t, std::map<uint32_t, uint32_t>> kp_match_index_map;

    bool match_exists(const uint32_t kp_index, const uint32_t img_index)
    { return kp_match_index_map[kp_index].count(img_index) > 0; }


    OfflineCamera(Eigen::Vector3d position, Eigen::Matrix3d rotation, 
        float scale, Eigen::Vector3d dt, Eigen::Matrix3d dR, 
        Eigen::Matrix4d T, Eigen::Matrix<double, 3, 4> P,
        std::vector<double> distortion_coeffs, std::shared_ptr<o3d::geometry::PointCloud> view_cloud,
        std::vector<cv::KeyPoint> kp_features, cv::Mat feature_descriptors,
        std::shared_ptr<o3d::geometry::Image> rgb, std::shared_ptr<o3d::geometry::Image> depth,
        std::shared_ptr<cv::Mat> cv_rgb, std::shared_ptr<cv::Mat> cv_depth, 
        o3d::camera::PinholeCameraIntrinsic intr, int id, float rodriques_angle, 
        std::string rgb_name, std::string depth_name, std::map<uint32_t, uint32_t> landmark_lookup,
        std::map<uint32_t, std::map<uint32_t, uint32_t>> kp_match_index_map) :

            Camera(position, rotation, scale, dt, dR, T, P,
            distortion_coeffs, view_cloud, kp_features, feature_descriptors,
            rgb, depth, cv_rgb, cv_depth, intr, id,
            rodriques_angle, rgb_name, depth_name),
            landmark_lookup(landmark_lookup), kp_match_index_map(kp_match_index_map)
        {

        }

    OfflineCamera() { }

};

}