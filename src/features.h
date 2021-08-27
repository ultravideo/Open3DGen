#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <tuple>
#include <thread>
#include "constants.h"
#include "timer.h"
#include "feature_graph.h"
#include "s3d_camera.h"
#include "rgbd_utilities.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/xfeatures2d.hpp>

namespace stitcher3d::features
{
    std::vector<Eigen::Vector2d> undistort_features(const std::vector<cv::KeyPoint>& features, 
        const std::vector<double>& dist_coeffs, const o3d::camera::PinholeCameraIntrinsic& intr);
    std::vector<Eigen::Vector2d> undistort_features(const std::vector<Eigen::Vector2d>& features, 
        const std::vector<double>& dist_coeffs, const o3d::camera::PinholeCameraIntrinsic& intr);

    Eigen::Vector2d undistort_feature(const Eigen::Vector2d& feature, const std::vector<double>& dist_coeffs, 
        const o3d::camera::PinholeCameraIntrinsic& intr);

    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_features(const std::shared_ptr<o3d::geometry::Image> rgb, const uint32_t max_feature_count,
        const o3d::camera::PinholeCameraIntrinsic* intr = nullptr, const std::vector<double>* dist_coeffs = nullptr);

    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_akaze_features(const std::shared_ptr<cv::Mat> rgb, 
        const o3d::camera::PinholeCameraIntrinsic* intr = nullptr, const std::vector<double>* dist_coeffs = nullptr);

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> match_features_no_homography(const std::vector<cv::KeyPoint>& kp1, 
        const cv::Mat& desc1, const std::vector<cv::KeyPoint>& kp2, const cv::Mat& desc2,
        const std::shared_ptr<o3d::geometry::Image> img1 = nullptr, const std::shared_ptr<o3d::geometry::Image> img2 = nullptr);

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> match_features(const std::vector<cv::KeyPoint>& kp1, 
        const cv::Mat& desc1, const std::vector<cv::KeyPoint>& kp2, const cv::Mat& desc2, 
        const std::shared_ptr<o3d::geometry::Image> img1 = nullptr, const std::shared_ptr<o3d::geometry::Image> img2 = nullptr);
    
    std::vector<cv::Point2f> matches_to_points(const std::vector<cv::KeyPoint>& matches);

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filter_matches_triangulation(Camera& cam0, Camera& cam1, 
        const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> rc_corr);

    void filter_triangulated_3view(std::vector<Eigen::Vector3d>& trpoints, std::vector<std::vector<feature_vertex>>& fvs,
        const std::vector<Camera*>& cameras);

    std::tuple<std::vector<cv::Mat>, std::vector<Eigen::Vector2i>> image_to_cells(const cv::Mat image, const int cell_size_x, const int cell_size_y);

    inline cv::Mat triangulate_Linear_LS(cv::Mat mat_P_l, cv::Mat mat_P_r, cv::Mat warped_back_l, cv::Mat warped_back_r)
    {
        cv::Mat A(4, 3, CV_64FC1), b(4, 1, CV_64FC1), X(3, 1, CV_64FC1), X_homogeneous(4, 1, CV_64FC1), W(1, 1, CV_64FC1);
        
        W.at<double>(0,0) = 1.0;
        A.at<double>(0,0) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,0) - mat_P_l.at<double>(0,0);
        A.at<double>(0,1) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,1) - mat_P_l.at<double>(0,1);
        A.at<double>(0,2) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,2) - mat_P_l.at<double>(0,2);
        A.at<double>(1,0) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,0) - mat_P_l.at<double>(1,0);
        A.at<double>(1,1) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,1) - mat_P_l.at<double>(1,1);
        A.at<double>(1,2) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,2) - mat_P_l.at<double>(1,2);
        A.at<double>(2,0) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,0) - mat_P_r.at<double>(0,0);
        A.at<double>(2,1) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,1) - mat_P_r.at<double>(0,1);
        A.at<double>(2,2) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,2) - mat_P_r.at<double>(0,2);
        A.at<double>(3,0) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,0) - mat_P_r.at<double>(1,0);
        A.at<double>(3,1) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,1) - mat_P_r.at<double>(1,1);
        A.at<double>(3,2) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,2) - mat_P_r.at<double>(1,2);
        b.at<double>(0,0) = -((warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,3) - mat_P_l.at<double>(0,3));
        b.at<double>(1,0) = -((warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,3) - mat_P_l.at<double>(1,3));
        b.at<double>(2,0) = -((warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,3) - mat_P_r.at<double>(0,3));
        b.at<double>(3,0) = -((warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,3) - mat_P_r.at<double>(1,3));

        cv::solve(A, b, X, cv::DECOMP_SVD);
        cv::vconcat(X, W, X_homogeneous);
        return X_homogeneous;
}

    /**
     * From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
     */
    inline cv::Mat_<double> linear_LS_Triangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1)
    {
        //build matrix A for homogenous equation system Ax = 0
        //assume X = (x,y,z,1), for Linear-LS method
        //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
        cv::Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
            u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
            u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
            u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
                );
        cv::Mat_<double> B = (cv::Mat_<double>(4, 1) <<    -(u.x*P(2,3)    -P(0,3)),
                        -(u.y*P(2,3)  -P(1,3)),
                        -(u1.x*P1(2,3)    -P1(0,3)),
                        -(u1.y*P1(2,3)    -P1(1,3)));
    
        cv::Mat_<double> X;
        solve(A, B, X, cv::DECOMP_SVD);
    
        return X;
    }

    std::vector<cv::Point2f> normalize_features(std::vector<cv::Point2f> points_vec);

    void draw_matches_indexes(const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1);
    
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filter_matches_radius(
        const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1,
        const float max_radius_ratio = FEATURE_MATCHES_OUTLIER_RADIUS_RATIO,
        const float min_radius_ratio = FEATURE_MATCHES_OUTLIER_RADIUS_RATIO_MIN);

    std::vector<std::pair<uint32_t, uint32_t>> filter_matches_radius(
        const std::vector<std::pair<uint32_t, uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1,
        const float max_radius_ratio = FEATURE_MATCHES_OUTLIER_RADIUS_RATIO,
        const float min_radius_ratio = FEATURE_MATCHES_OUTLIER_RADIUS_RATIO_MIN);

}
