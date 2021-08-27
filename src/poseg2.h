/**
 *   INITIAL
 *   - initial with essential
 *   - all matches to landmarks
 *       - triangulate
 *   - all unmatched to candidates
 *   RECURRING
 *   1. Camera Pose Estimation
 *       - detect features
 *       - match features against landmarks
 *           - lm average descriptor
 *       - return vector<landmark_id, feature_id>
 *       - create 3d-2d correspondences
 *           - lm average 3d point
 *       - PnP
 *   2. Landmarks and Candidates
 *       - append matched to landmarks
 *       - create current view feature negative (all unmatched features)
 *       - match negative against candidates
 *           - return vector<candidate_id, feature_id>
 *       - add to landmarks from candidate matches
 *       - create new landmarks from candidate matches
 *       - remove matched candidates
 *       - add all remaining features to candidates
 */


#pragma once

#include <cstdint>
#include <iostream>
#include <Open3D/Open3D.h>
#include <Open3D/Registration/ColoredICP.h>
#include <opencv2/core/types.hpp>
#include <set>
#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include <algorithm>
#include <thread>
#include <mutex>
#include <cmath>

#include "constants.h"
#include "rgbd_utilities.h"
#include "s3d_camera.h"
#include "math_modules.h"
#include "rgbd_utilities.h"
#include "pose_predictor.h"

#include "feature_graph.h"

#include "features.h"
#include "registration_params.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h> 


namespace stitcher3d
{

struct FrameFragment
{
    std::vector<Camera*> frag_cameras;
    std::map<uint32_t, std::vector<uint32_t>> frag_feature_tracks;
    std::vector<Eigen::Matrix4d> frag_frame_transforms;
    std::vector<cv::Point3f> feature_3d_points;
};

/**
 *  A landmark is a point in 3D space, whose position
 *      is already inferred
 */
struct Landmark
{
    struct ViewData
    {
        uint32_t camera_id;
        uint32_t feature_id;
        cv::KeyPoint feature_kp;
        cv::Mat feature_descriptor;
        cv::Mat camera_P;
    };

    std::pair<uint32_t, uint32_t> triangulation_view_pair;

    std::vector<ViewData> view_data;
    cv::Mat view_average_descriptor;
    cv::Point3f point3d = cv::Point3f(0.f, 0.f, 0.f);
    Eigen::Vector3d point_color;

    cv::Point3f sum_point = cv::Point3f(0.0f, 0.0f, 0.0f);
    uint32_t sum_count = 0;

    cv::Mat compute_view_average_descriptor() const
    {
        if (view_data.size() == 0)
            throw std::runtime_error("no views found, aborting!");

        cv::Mat view_avg_desciptor;

        // handle special case where there's only 1 view e.g. candidates
        if (view_data.size() == 1)
        {
            view_avg_desciptor = view_data[0].feature_descriptor;
            return view_avg_desciptor;
        }

        /**
         *  Create an average descriptor for this frame
         */
        const int descriptor_cols = view_data.at(0).feature_descriptor.cols;
        cv::Mat temp_desc = cv::Mat::zeros(1, descriptor_cols, CV_32F);

        for (int i = 0; i < view_data.size(); i++)
        {
            for (int j = 0; j < descriptor_cols; j++)
            {
                temp_desc.at<float>(0, j) += (float)view_data[i].feature_descriptor.at<float>(0, j);
            }
        }

        view_avg_desciptor = cv::Mat::zeros(1, descriptor_cols, CV_32F);
        for (int i = 0; i < descriptor_cols; i++)
        {
            view_avg_desciptor.at<float>(0, i) = (float)(temp_desc.at<int>(0, i) / (float)view_data.size());
        }

        return view_avg_desciptor;
    }


    /** 
     *  TODO:
     *      keep a sum_3d_point in memory and only add to it,
     *          in get_3d divide by size? Performance impact?
     */
    

    void add_view(const uint32_t cam_id, const uint32_t feature_id,
        const cv::KeyPoint feature_kp, const cv::Mat descriptor,
        const cv::Mat camera_P, 
        const std::vector<Camera*>& cameras = std::vector<Camera*>(),
        const double lm_avg_magnitude = 0.0,
        const bool do_triangulate = false)
    {
        const ViewData cvd
        {
            cam_id,
            feature_id,
            feature_kp,
            descriptor,
            camera_P
        };

        view_data.emplace_back(cvd);
        // compute_view_average_descriptor();

        if (do_triangulate)
        {

            if (sum_count == 0)
            {
                retriangulate_full(cameras, lm_avg_magnitude);
                return;
            }
            /**
             *  try-triangulate the new view against all the previous ones
             */
            const uint32_t ref_vd_id = view_data.size() - 1;

            
            std::vector<cv::Point3f> points;
            const Camera* last = cameras.at(view_data.back().camera_id);

            for (int i = 0; i < view_data.size() - 1; i++)
            {
                const Camera* first = cameras.at(view_data[i].camera_id);

                // enough movement for triangulation
                if ((last->position - first->position).norm() > CANDIDATE_TRANSLATION_THRESHOLD ||
                    math::rad2deg((math::eigen_rot2euler(last->rotation) - math::eigen_rot2euler(first->rotation)).norm()) > CANDIDATE_ROTATION_THRESHOLD)
                {
                    // no need to verify the triangulation, as the triangulation pair should already be valid
                    const cv::Point3f tr = this->get_triangulated(i, ref_vd_id);

                    // don't add if triangulated is outlier (e.g. magnitude too great or behind cameras)
                    if ((lm_avg_magnitude > 0.0 && (sqrt(tr.dot(tr)) > lm_avg_magnitude * LM_TRIANGULATION_DISTANCE_OUTLIER_MULTIPLIER))
                     || !(math::point_infront_of_camera(last->P, Eigen::Vector3d(tr.x, tr.y, tr.z)) && math::point_infront_of_camera(first->P, Eigen::Vector3d(tr.x, tr.y, tr.z))))
                    {
                        continue;
                    }

                    points.push_back(tr);
                }
                // not enough movement
                else { continue; }
                            
            }


            if (points.size() == 0)
            {
                return;
            }

            // average the point
            for (const auto& p : points)
                sum_point += p;
                // sum_point += cv::Point3d(p.x, p.y, p.z);

            sum_count += (uint32_t)points.size();

            const cv::Point3f avg_3d = sum_point / (double)sum_count;
            this->point3d = avg_3d;
        }
    }

    /**
     *  Only adds the required (minimal) information,
     *  used in the "slow" -reconstruction
     */
    void add_view_simple(const uint32_t cam_id, const uint32_t feature_id, const cv::Point3f p3d)
    {
        const ViewData cvd
        {
            cam_id,
            feature_id,
            cv::KeyPoint(),
            cv::Mat(),
            cv::Mat()
        };

        view_data.emplace_back(cvd);
        sum_point += cv::Point3f(p3d.x, p3d.y, p3d.z);
    }

    void set_3d_point(const cv::Point3f p3d)
    {
        point3d = p3d;
    }

    void add_as_candidate(const uint32_t cam_id, const uint32_t feature_id, 
        const cv::KeyPoint feature_kp, const cv::Mat descriptor, const cv::Mat camera_P)
    {
        const ViewData cvd
        {
            cam_id,
            feature_id,
            feature_kp,
            descriptor,
            camera_P
        };

        view_data.emplace_back(cvd);
    }

    void avg_3d_position()
    {
        point3d /= (double)(view_data.size() - 1);
    }

    /**
     *  triangulates and converts a candidate to a landmark.
     *  returns false on failure
     */
    void to_landmark(const std::vector<Camera*>& cameras)
    {
        const int x = view_data.front().feature_kp.pt.x;
        const int y = view_data.front().feature_kp.pt.y;

        point_color =
            Eigen::Vector3d(
                (double)(*cameras.at(view_data.front().camera_id)->rgb->PointerAt<unsigned char>(x, y, 0)) / 255.0,
                (double)(*cameras.at(view_data.front().camera_id)->rgb->PointerAt<unsigned char>(x, y, 1)) / 255.0,
                (double)(*cameras.at(view_data.front().camera_id)->rgb->PointerAt<unsigned char>(x, y, 2)) / 255.0
            );
    }

    void retriangulate_full(const std::vector<Camera*>& cameras, double lm_avg_magnitude = 0.0)
    {
        /**
         *  pairwise triangulate with all pairs which have enough movement,
         *  take average of 3d point
         * 
         *  NOTE:
         *      add "triangulate new", only pairwise from newest view (in add_view?)
         *          until not enough movement --> break.
         *      to_landmark fully retriangulates the entire thing (?)
         *  
         *  TODO:
         *      in front of camera here
         *      https://github.com/libmv/libmv/blob/8040c0f6fa8e03547fd4fbfdfaf6d8ffd5d1988b/src/libmv/multiview/projection.h#L183
         */

        std::vector<cv::Point3f> points;

        for (int i = 0; i < view_data.size() - 1; i++)
        {
            for (int j = i + 1; j < view_data.size(); j++)
            {
                const Camera* first = cameras.at(view_data[i].camera_id);
                const Camera* last = cameras.at(view_data[j].camera_id);

                // enough movement for triangulation
                if ((last->position - first->position).norm() > CANDIDATE_TRANSLATION_THRESHOLD ||
                    math::rad2deg((math::eigen_rot2euler(last->rotation) - math::eigen_rot2euler(first->rotation)).norm()) > CANDIDATE_ROTATION_THRESHOLD)
                {
                    // no need to verify the triangulation, as the triangulation pair should already be valid
                    const cv::Point3f tr = this->get_triangulated(i, j);

                
                    /**
                     *  NOTE:
                     *      having the in-front-of-camera -check here _may_ make some sequences fail
                     */

                    // don't add if triangulated is outlier (e.g. magnitude too great or behind cameras)
                    if ((lm_avg_magnitude > 0.0 && (sqrt(tr.dot(tr)) > lm_avg_magnitude * LM_TRIANGULATION_DISTANCE_OUTLIER_MULTIPLIER))
                        || !(math::point_infront_of_camera(last->P, Eigen::Vector3d(tr.x, tr.y, tr.z)) && math::point_infront_of_camera(first->P, Eigen::Vector3d(tr.x, tr.y, tr.z)))
                    )
                    {
                        continue;
                    }

                    points.push_back(tr);
                }
                // not enough movement -> stop triangulation, as indexing is "pinching", 
                // no further opporunity for triangulation
                else { continue; }
            }
        }

        if (points.size() == 0)
        {
            return;
        }

        // average the point
        cv::Point3f avg_3d;
        for (const auto& p : points)
            avg_3d += p;

        this->sum_point = avg_3d;
        this->sum_count = (uint32_t)points.size();

        avg_3d = avg_3d / (double)points.size();
        this->point3d = avg_3d;
    }

    bool set_triangulate(const uint32_t first_view_id = 0, int second_view_id = -1)
    {
        if (second_view_id == -1)
            second_view_id = view_data.size() - 1;

        const cv::Point3f p3d = get_triangulated(first_view_id, (uint32_t)second_view_id);

        if (sqrt(p3d.dot(p3d)) > TRIANGULATED_POINT_OUTLIER_NORM)
            return false;

        point3d = p3d;

        return true;
    }

    cv::Point3f get_triangulated(const uint32_t first_view_id, const uint32_t second_view_id)
    {
        const std::vector<cv::Point2f> fkp1 = { view_data[first_view_id].feature_kp.pt };
        const std::vector<cv::Point2f> fkp2 = { view_data[second_view_id].feature_kp.pt };

        // std::cout << "point coords: " << fkp1 << ", " << fkp2 << "\n";

        const cv::Mat P1 = view_data[first_view_id].camera_P;
        const cv::Mat P2 = view_data[second_view_id].camera_P;

        cv::Mat p4d;
        cv::triangulatePoints(P1, P2, fkp1, fkp2, p4d);

        const cv::Point3f p3d(
            p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
            p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
            p4d.at<float>(2, 0) / p4d.at<float>(3, 0)
        );

        return p3d;
    }
};

/**
 * The translation between cameras 0-1 will always be 1.0f, everything will be scaled according to that.
 */
class PoseGraph
{
public:
    PoseGraph();
    ~PoseGraph();

    Camera* add_camera(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation,
        const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& ,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const std::string rgb_name, const std::string depth_name, const bool offline = false);

    /**
     *  computes a world pose for the given camera.
     *  requires at least two cameras to be already in
     *  the graph
     */
    bool compute_pose_realtime(Camera* cam, const uint32_t time_id);

    /**
     *  computes a relative pose between two cameras. 
     *  should only be used for the first two cameras 
     *  in the graph
     */
    bool compute_initial_pose(Camera* cam, const Camera* ref_cam);

    /**
     *  adds a new view into the posegraph in real teim
     */
    bool add_camera_and_compute_pose_realtime(const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& depth,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const uint32_t time_id, const std::string rgb_name, const std::string depth_name);

    /**
     *  for accurate offline camera pose estimation
     */
    void add_camera_offline(const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& depth,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const std::string rgb_name, const std::string depth_name);

    /**
     *  computes the camera poses pair-wise, more accurate
     *  but slower.
     */
    void compute_camera_poses_accurate();

    /**
     *  shows the cameras, triangulates points and depthclouds in a 3d view
     */
    void visualize_tracks() const;

    static inline Eigen::Matrix4d compose_transform_from_Rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
    {
        Eigen::Matrix4d transform;
        Eigen::Matrix<double, 3, 4> temp43;
        temp43 << R, t;
        transform << temp43, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0).transpose();
        return transform;
    }


    double get_pointcloud_scale() const
    {
        return pointcloud_scale;
    }

    void set_registration_parameters(const registration::RegistrationParams& rparams) { reg_params = rparams; }

    // static std::vector<Eigen::Vector3d> triangulate_matches_world(const Camera* cam0, const Camera* cam1, 
    //     const std::pair<std::vector<uint32_t>, std::vector<uint32_t>>& matches);

    static inline Eigen::Vector3d transform_point_R_t(const Eigen::Vector3d& point3d, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
    {
        const Eigen::Matrix4d transform = compose_transform_from_Rt(R, t);
        const Eigen::Vector4d point (point3d.x(), point3d.y(), point3d.z(), 1.0);
        const Eigen::Vector4d transformed_point = transform * point;
        return Eigen::Vector3d(transformed_point.x(), transformed_point.y(), transformed_point.z());
    }
    
    static inline void set_camera_T_and_P(const Camera* refcam, Camera* cam)
    {
        throw std::runtime_error("not tested!");

        const Eigen::Matrix3d& dR = cam->dR;
        const Eigen::Vector3d& dt = cam->dt;

        const Eigen::Matrix4d T = compose_transform_from_Rt(dR, dt);
        cam->T = refcam->T * T;

        const Eigen::Vector4d t_temp = cam->T.col(3);
        const Eigen::Matrix3d R = cam->T.block(0, 0, 3, 3);
        const Eigen::Vector3d t (t_temp.x(), t_temp.y(), t_temp.z());

        cam->rotation = R;
        cam->position = t;

        Eigen::Matrix<double, 3, 4> P;
        // P.block(0, 0, 3, 3) = R.transpose();
        // P.col(3) = -R.transpose() * t;
        // P = cam->intr.intrinsic_matrix_ * P;
        math::P_From_KRt(cam->intr.intrinsic_matrix_, R.transpose(), -R.transpose() * t, &P);

        cam->P = P;

        const Eigen::AngleAxisd ax(cam->rotation);
        cam->rodriques_angle = ax.angle() * (180.0/3.14159);
    }

    static inline void set_camera_T_and_P(Camera* cam, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const bool transpose_flip = true)
    {
        // const Eigen::Matrix3d Rt = R.transpose();
        // const Eigen::Vector3d tt = -Rt * t;
        const Eigen::Matrix4d T = compose_transform_from_Rt(R, t);
        cam->T = T;

        cam->rotation = R;
        cam->position = t;

        Eigen::Matrix<double, 3, 4> P;
        if (transpose_flip)
            math::P_From_KRt(cam->intr.intrinsic_matrix_, R.transpose(), -R.transpose() * t, &P);
        else
            math::P_From_KRt(cam->intr.intrinsic_matrix_, R, t, &P);

        cam->P = P;

        const Eigen::AngleAxisd ax(cam->rotation);
        cam->rodriques_angle = ax.angle() * (180.0/3.14159);
    }

    static inline void set_camera_T_and_P(Camera* cam, const Eigen::Matrix4d new_T)
    {
        cam->T = new_T;

        const Eigen::Vector4d t_temp = cam->T.col(3);
        const Eigen::Matrix3d R = cam->T.block(0, 0, 3, 3);
        const Eigen::Vector3d t (t_temp.x(), t_temp.y(), t_temp.z());

        cam->rotation = R;
        cam->position = t;

        Eigen::Matrix<double, 3, 4> P;
        math::P_From_KRt(cam->intr.intrinsic_matrix_, R.transpose(), -R.transpose() * t, &P);

        cam->P = P;

        const Eigen::AngleAxisd ax(cam->rotation);
        cam->rodriques_angle = ax.angle() * (180.0/3.14159);
    }

    void recompute_camera_poses_landmark_PnP();

    /**
     *  removes a camera
     *  WARNING: does not remove camera from
     *      landmarks or candidates!
     */
    void remove_last_camera(Camera* cam);

    void create_pointclouds_from_depths(const bool only_scale = false);

    std::vector<std::shared_ptr<Camera>> get_cameras_copy() const;

    /**
     *  calculates the relative translation + rotation between two views
     *  using the matched 2d features
     */
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d> relative_pose_for_camera(const Camera* ref_cam, Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> feature_matches);

    /**
     *  triangulates a pointcloud from the current held landmarks
     */
    std::shared_ptr<o3d::geometry::PointCloud> landmarks_to_pointcloud() const;

    void retriangulate_landmarks();

    // <lms, feature>
    void retriangulate_landmarks(const std::vector<std::pair<uint32_t, uint32_t>> landmark_features);

    void bundle_adjust();

    std::vector<Camera*> cameras;

    std::vector<std::shared_ptr<o3d::geometry::PointCloud>> depth_pointclouds;


    /**
     *  contains data about tracked points
     */
    std::vector<Landmark> landmarks;

    /**
     *  contains 2d features that were detectde, but not matched.
     *  The current view features, which were not matched against landmarks,
     *  will be added here after evey succesfull frame. Additionally, after every frame,
     *  this list will be tried against the current view unmatched and made to landmarks, 
     *  if possible.
     */
    std::vector<Landmark> candidates;

private:

    /**
     *  refines the cam's pose with ICP pointcloud registration.
     *  Requires depth data for both cameras to be valid.
     *  Assumes ref_cam is perfectly positioned in the world.
     *  Sets the pointcloud scale to be 1.0
     */
    void refine_pose_with_ICP(Camera* cam, const Camera* ref_cam);

    /**
     *  creates a descriptor from all landmarks
     *  view_offset specifies how many frames from back
     *  are taken
     */
    cv::Mat create_landmark_descriptor(const uint32_t view_offset = 0) const;

    /**
     *  creates a descriptor from all candidates, last_valid
     *  is the biggest camera index to be considered
     */
    cv::Mat create_candidate_descriptor() const;

    /**
     *  matches two descriptors against each other using a brute force matcher
     */
    std::unique_ptr<cv::BFMatcher> f_matcher_crosscheck, f_matcher;
    inline std::vector<std::pair<uint32_t, uint32_t>>
        match_features_bf(const cv::Mat fdesc1, const cv::Mat fdesc2, 
        const float MAX_DIST = MAX_MATCH_DISTANCE) const
    {
        // std::cout << fdesc1 << "\n";
        if (fdesc1.rows < MIN_FEATURE_MATCH_COUNT && fdesc2.rows < MIN_FEATURE_MATCH_COUNT)
            throw std::runtime_error("not enough feature matches, aborting!");

        std::vector<cv::DMatch> nn_matches;
        // f_matcher_crosscheck->knnMatch(fdesc1, fdesc2, nn_matches, 1);
        f_matcher_crosscheck->match(fdesc1, fdesc2, nn_matches);

        std::vector<std::pair<uint32_t, uint32_t>> inliers;
        for (int i = 0; i < nn_matches.size(); i++)
        {
            const float dist1 = nn_matches[i].distance;

            if (dist1 > MAX_DIST)
                continue;

            inliers.emplace_back(std::make_pair(nn_matches[i].queryIdx, nn_matches[i].trainIdx));
        }
        return inliers;
    }

    /**
     *  uses distance check
     */
    inline std::vector<std::pair<uint32_t, uint32_t>>
        match_features_bf_knn(const cv::Mat& fdesc1, const cv::Mat& fdesc2) const
    {
        if (fdesc1.rows < MIN_FEATURE_MATCH_COUNT && fdesc2.rows < MIN_FEATURE_MATCH_COUNT)
            throw std::runtime_error("not enough feature matches, aborting!");

        std::vector<std::vector<cv::DMatch>> nn_matches;
        f_matcher->knnMatch(fdesc1, fdesc2, nn_matches, 2);

        std::vector<std::pair<uint32_t, uint32_t>> inliers;

        for(size_t i = 0; i < nn_matches.size(); i++)
        {
            const cv::DMatch first = nn_matches[i][0];
            const float dist1 = nn_matches[i][0].distance;
            const float dist2 = nn_matches[i][1].distance;

            if (dist1 < NN_MATCH_RATIO * dist2)
            {               
                inliers.emplace_back(std::make_pair(first.queryIdx, first.trainIdx));
            }
        }

        return inliers;
    }

    /**
     *  matches features using homography to filter outliers
     */
    inline std::vector<std::pair<uint32_t, uint32_t>>
        match_features_hgraphy(const cv::Mat fdesc1, const cv::Mat fdesc2,
        const Camera* cam1, const Camera* cam2,
        const float threshold = INLIER_THRESHOLD) const
    {
        if (fdesc1.rows < MIN_FEATURE_MATCH_COUNT && fdesc2.rows < MIN_FEATURE_MATCH_COUNT)
            return std::vector<std::pair<uint32_t, uint32_t>>();
            // throw std::runtime_error("not enough feature matches, aborting!");

        // match the descriptors using crosscheck

        std::vector<cv::DMatch> nn_matches;
        f_matcher_crosscheck->match(fdesc1, fdesc2, nn_matches);

        std::vector<std::pair<uint32_t, uint32_t>> inliers, ret_inliers;
        std::vector<cv::Point2f> matched1, matched2;

        for (int i = 0; i < nn_matches.size(); i++)
        {
            inliers.emplace_back(std::make_pair(nn_matches[i].queryIdx, nn_matches[i].trainIdx));

            matched1.push_back(cam1->kp_features[nn_matches[i].queryIdx].pt);
            matched2.push_back(cam2->kp_features[nn_matches[i].trainIdx].pt);
        }

        ret_inliers.reserve(inliers.size());


        // filter outliers with homography check

        cv::Mat inlier_mask, homography;
        std::vector<cv::DMatch> inlier_matches;

        if (matched1.size() >= 4)
        {
            homography = findHomography(matched1, matched2, cv::RANSAC, ransac_thresh, inlier_mask);
        }
        else
            return std::vector<std::pair<uint32_t, uint32_t>>();
            // throw std::runtime_error("not enough matched points, aborting in match_features!");


        for (size_t i = 0; i < inliers.size(); i++)
        {
            cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = matched1[i].x;
            col.at<double>(1) = matched1[i].y;
            col = homography * col;
            col /= col.at<double>(2);

            const double dist = sqrt(pow(col.at<double>(0) - matched2[i].x, 2) + pow(col.at<double>(1) - matched2[i].y, 2));
            if (dist < INLIER_THRESHOLD)
            {
                ret_inliers.emplace_back(inliers[i]);
            }
        }

        ret_inliers.shrink_to_fit();
        return ret_inliers;
    }

    /**
     *  [x_dir, y_dir], where positive values indicate right and up
     *  the larger the magnitude, the better quality feature matches
     */
    inline static std::pair<float, float> feature_average_translation(
        const std::vector<cv::Point2f> x1v, const std::vector<cv::Point2f> x2v)
    {
        if (x1v.size() != x2v.size())
            throw std::runtime_error("feature vector sizes didn't match, aborting!");

        float x = 0.0f;
        float y = 0.0f;

        for (int i = 0; i < x1v.size(); i++)
        {
            const cv::Point2f x1 = x1v[i];
            const cv::Point2f x2 = x2v[i];

            if (abs(x2.x - x1.x) < INPRECISE_FEATURE_THRESHOLD)
            {
                // do nothing, or "add zero"
            }
            else if (x2.x > x1.x)
            {
                x += 1.0f;
            }
            else
            {
                x -= 1.0f;
            }

            if (abs(x2.y - x1.y) < INPRECISE_FEATURE_THRESHOLD)
            {
                // do nothing, or "add zero"
            }
            else if (x2.y > x1.y)
            {
                y += 1.0f;
            }
            else
            {
                y -= 1.0f;
            }
        }

        return std::make_pair(x, y);
    }

    /**
     *  returns the indexes of points which were not behind the camera, e.g. valid
     */
    std::vector<uint32_t> filter_points_behind_camera(const Camera* cam, const std::vector<cv::Point3f> points3d) const;

    /**
     *  creates an inverse vector from matched <landmark_id, feature_id>
     *  containing only the unmatched feature indexes. assumes the features
     *  are originally linear and contain all numbers between [0, all_features_count[
     */
    std::vector<uint32_t> create_match_negative(
        const std::vector<std::pair<uint32_t, uint32_t>> matched_features,
        const uint32_t all_features_count) const;

    /**
     *  adds matched <landmark_id, feature_id> to the landmarks.
     *  does not create new landmarks
     */
    void add_track_to_landmarks(const Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view,
        const std::vector<cv::Point3f> points3d,
        const std::vector<int> inliers);

    std::vector<uint32_t> set_negative_intersection(const std::vector<uint32_t> subset, const std::vector<uint32_t> all) const;

    /**
     *  matches negative list features against candidates
     *  return vector<candidate_id, feature_id>
     */
    std::vector<std::pair<uint32_t, uint32_t>> match_features_against_candidates(const Camera* cam,
        const std::vector<uint32_t> matches_negative,
        std::map<std::pair<uint32_t, uint32_t>, cv::Mat>& camera_hgraphy_pairs) const;

    /**
     *  filters outliers in candidate matches with homography.
     *  returns <candidate_id, feature_id>
     */
    std::vector<std::pair<uint32_t, uint32_t>> filter_candidate_matches_homography(
        const std::vector<std::pair<uint32_t, uint32_t>> unfiltered,
        const Camera* cam);

    /**
     *  creates new landmarks from matched candidates
     *  <candidate_id, feature_id>
     *  returns a list of succesfully added candidate ids
     */
    std::vector<uint32_t> add_landmarks_from_candidates(const Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> candidate_matches);

    /**
     *  matches the current view against all landmarks using 
     *  landmark average descriptor
     *  returns <landmark_id, feature_id>
     */
    std::vector<std::pair<uint32_t, uint32_t>> find_landmarks_in_view(const Camera* cam, 
        std::map<std::pair<uint32_t, uint32_t>, cv::Mat>& camera_hgraphy_pairs,
        const uint32_t view_offset = 0) const;

    /**
     *  matches: <feature_id, feature_id>
     */
    std::vector<cv::Point3f> triangulate_matches(
        const std::vector<std::pair<uint32_t, uint32_t>> matches,
        const std::vector<uint32_t> first_cams, const Camera* second_cam) const;

    /**
     *  creates landmarks from the given feature matches and the 
     *  corresponding cameras. 
     */
    void landmarks_from_feature_matches(
        const std::vector<std::pair<uint32_t, uint32_t>> matches,
        const Camera* first_cam, const Camera* second_cam);
    
    /**
     *  creates candidates from features that were not matched at all
     */
    void candidates_from_unmatched(const std::vector<uint32_t> unmatched_features, const Camera* cam);

    void remove_candidates(std::vector<uint32_t> matched_candidates);

    /**
     *  a more sophisticated version of find_match_negative
     */
    std::vector<uint32_t> find_negative_features_from_unmatched(
            const std::vector<std::pair<uint32_t, uint32_t>> feature_candidate_matches,
            const std::vector<uint32_t> landmark_unmatched_features) const;

    /**
     *  creates descriptors from given feature ids
     */
    inline cv::Mat descriptors_from_features(const Camera* cam, const std::vector<uint32_t> feature_ids) const
    {
        if (feature_ids.size() > cam->feature_descriptors.rows)
        {
            std::cout << "camera feature_descriptors: " << cam->feature_descriptors.rows << ", " << "feature_ids: " << feature_ids.size() << "\n";
            throw std::runtime_error("feature_ids size was smaller than feature_Descriptors size, aborting!");
        }

        cv::Mat descriptor (feature_ids.size(), cam->feature_descriptors.cols, CV_32F);

        // set the descriptor rows to be landmark descriptors
        for (int i = 0; i < feature_ids.size(); i++)
        {
            cv::Mat feature_descriptor;
            feature_descriptor = cam->feature_descriptors.row(feature_ids[i]);
            const float* fdptr = feature_descriptor.ptr<float>(0, 0);
            
            float* fptr = descriptor.ptr<float>(i, 0);
            for (int j = 0; j < descriptor.cols; j++)
            {
                fptr[j] = fdptr[j];
            }
        }

        return descriptor;
    }

    static void DEBUG_visualize_features(const std::shared_ptr<o3d::geometry::Image> img, const std::vector<cv::Point2f> features)
    {
        const cv::Mat imgcv = rgbd::o3d_image_to_cv_image(img);

        float h_i = 0.0f;
        const float increment = 360.0f / (float)features.size();
        for (int i = 0; i < features.size(); i++)
        {
            const cv::Point2f kp = features[i];

            const auto [r, g, b] = rgbd::HSL_to_RGB(h_i, 90.0f + (float)rand() / (float)RAND_MAX * 10.0f, 50.0f + (float)rand() / (float)RAND_MAX * 10.0f);
            const cv::Scalar rand_color (r, g, b);
            h_i += increment;

            cv::circle(imgcv, cv::Point(kp.x, kp.y), 6, rand_color, 3);
        }

        cv::imshow("debug_features", imgcv);
        cv::waitKey(0);
    }

    static void DEBUG_visualize_matches(const Camera* ref_cam, 
        const Camera* cam, const std::vector<std::pair<uint32_t, uint32_t>> feature_matches,
        const int time_on_screen = 0)
    {
        std::vector<cv::KeyPoint> fkp1, fkp2;
        std::vector<cv::DMatch> matches;

        for (int i = 0; i < feature_matches.size(); i++)
        {
            fkp1.push_back(ref_cam->kp_features[feature_matches[i].first]);
            fkp2.push_back(cam->kp_features[feature_matches[i].second]);
            matches.emplace_back(cv::DMatch(i, i, 0));
        }

        cv::Mat matchimg;
        cv::drawMatches(rgbd::o3d_image_to_cv_image(ref_cam->rgb), fkp1, rgbd::o3d_image_to_cv_image(cam->rgb), fkp2, matches, matchimg);
        cv::cvtColor(matchimg, matchimg, cv::COLOR_BGR2RGB);
        cv::imshow("img", matchimg);
        cv::waitKey(time_on_screen);
    }

    void DEBUG_visualize_landmarks(const std::vector<cv::Point3f> points) const;

    void DEBUG_visualize_landmarks(const std::vector<uint32_t> lms) const;

    // <lm_id, feature_id>
    void DEBUG_visualize_landmark_matches(const Camera* cam, const std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view) const
    {
        // collect the last one, if matches. Should be good enough for a brief look
        const uint32_t ref_id = landmarks[landmarks_in_view.front().first].view_data.back().camera_id;
        std::vector<std::pair<uint32_t, uint32_t>> feature_matches;
        feature_matches.reserve(landmarks_in_view.size());

        for (auto lmf : landmarks_in_view)
        {
            if (landmarks[lmf.first].view_data.back().camera_id == ref_id)
                feature_matches.push_back(std::make_pair(landmarks[lmf.first].view_data.back().feature_id, lmf.second));
        }

        std::cout << "visualized matches: " << feature_matches.size() << ", all lm matches: " << landmarks_in_view.size() << "\n";
        DEBUG_visualize_matches(cameras.at(ref_id), cam, feature_matches);

    }
    
    void DEBUG_visualize_landmark_matches_against_last(
        const Camera* cam, const std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view,
        const int show_on_screen = 0) const
    {
        const uint32_t ref_id = cam->id - 1;
        std::vector<std::pair<uint32_t, uint32_t>> feature_matches;
        feature_matches.reserve(landmarks_in_view.size());

        for (auto lmf : landmarks_in_view)
        {
            if (landmarks[lmf.first].view_data.back().camera_id == ref_id)
                feature_matches.push_back(std::make_pair(landmarks[lmf.first].view_data.back().feature_id, lmf.second));
        }

        std::cout << "visualized matches: " << feature_matches.size() << ", all lm matches: " << landmarks_in_view.size() << "\n";
        DEBUG_visualize_matches(cameras.at(ref_id), cam, feature_matches, show_on_screen);
    }

    inline void DEBUG_visualize_feature_matches_custom(const Camera* ref_cam, 
        const Camera* cam, const std::vector<std::pair<uint32_t, uint32_t>> feature_matches,
        const int time_on_screen = 0) const
    {
        cv::Mat himg;
        const cv::Mat img0 = rgbd::o3d_image_to_cv_image(ref_cam->rgb);
        const cv::Mat img1 = rgbd::o3d_image_to_cv_image(cam->rgb);

        const int width = img0.cols;

        cv::hconcat(img0, img1, himg);
        cv::cvtColor(himg, himg, cv::COLOR_BGR2RGB);

        srand(time(NULL));

        float h_i = 0.0f;
        const float increment = 360.0f / (float)feature_matches.size();
        for (int i = 0; i < feature_matches.size(); i++)
        {
            const cv::Point2f kp0 = ref_cam->kp_features[feature_matches[i].first].pt;
            const cv::Point2f kp1 = cam->kp_features[feature_matches[i].second].pt;

            const auto [r, g, b] = rgbd::HSL_to_RGB(h_i, 3.0f + (float)rand() / (float)RAND_MAX * 1.0f, 0.0f + (float)rand() / (float)RAND_MAX * 1.0f);
            const cv::Scalar rand_color (r, g, b);
            h_i += increment;

            cv::circle(himg, cv::Point(kp0.x, kp0.y), 3, rand_color);
            cv::circle(himg, cv::Point(kp1.x + width, kp1.y), 3, rand_color);

            cv::line(himg, cv::Point(kp0.x, kp0.y), cv::Point(kp1.x + width, kp1.y), rand_color, 2);
        }

        cv::imshow("img", himg);
        cv::waitKey(time_on_screen);

    }

    // <candidate id, current cam feature id>
    void DEBUG_visualize_candidate_matches(const Camera* cam, const std::vector<std::pair<uint32_t, uint32_t>> candidates_in_view, 
        const int time_on_screen = 0) const
    {
        if (candidates_in_view.size() == 0)
            return;

        // collect the last one, if matches. Should be good enough for a brief look
        const uint32_t ref_id = candidates[candidates_in_view.front().first].view_data.back().camera_id;
        std::vector<std::pair<uint32_t, uint32_t>> feature_matches;
        feature_matches.reserve(candidates_in_view.size());

        for (auto lmf : candidates_in_view)
        {
            if (candidates[lmf.first].view_data.back().camera_id == ref_id)
                feature_matches.push_back(std::make_pair(candidates[lmf.first].view_data.back().feature_id, lmf.second));
        }

        feature_matches.shrink_to_fit();

        std::cout << "visualized matches: " << feature_matches.size() << ", all candidate matches: " << candidates_in_view.size() << "\n";
        DEBUG_visualize_matches(cameras.at(ref_id), cam, feature_matches, time_on_screen);
    }

    /**
     *  adds matches to candidates, where applicable,
     *  creates landmarks if possible and removes unsuccesful candidates
     */
    void handle_feature_candidate_matches(const std::vector<std::pair<uint32_t, uint32_t>> matches, const Camera* cam, const bool check_for_lm = false);
    
    /**
     *  removes candidates that are too old to be considered valid
     */
    void remove_old_candidates(const uint32_t cam_id);

    inline void DEBUG_visualize_majority_candidate_matches(const std::vector<uint32_t> debug_landmarks,
        const uint32_t view_time = 0, const bool view_3d = true) const
    {
        if (debug_landmarks.size() == 0)
            return;

        std::map<uint32_t, std::vector<uint32_t>> camera_ids;

        // find the camera with the most features
        for (const uint32_t i : debug_landmarks)
        {
            const uint32_t cam_id = landmarks[i].view_data.back().camera_id - 1;
            
            if (camera_ids.find(cam_id) == camera_ids.end())
            {
                camera_ids.insert(std::make_pair(cam_id, std::vector<uint32_t> { i }));
            } else {
                camera_ids.at(cam_id).push_back(i);
            }
        }

        uint32_t max_cam_count = 0;
        uint32_t majority_cam_id = 0;
        for (const std::pair<uint32_t, std::vector<uint32_t>>& p : camera_ids)
        {
            if (p.second.size() > max_cam_count)
            {
                max_cam_count = p.second.size();
                majority_cam_id = p.first;
            }
        }

        const std::vector<uint32_t> majority_cam_landmarks = camera_ids.at(majority_cam_id);
        const Camera* ref_cam = cameras.at(majority_cam_id);
        const Camera* cam = cameras.at(landmarks[debug_landmarks[0]].view_data.back().camera_id);

        std::vector<Eigen::Vector3d> debug_points3d;
        // std::vector<cv::Point2f> debug_points2d1, debug_points2d2;
        std::vector<std::pair<uint32_t, uint32_t>> feature_matches;

        for (int i = 0; i < majority_cam_landmarks.size(); i++)
        {
            const Landmark& lm = landmarks[majority_cam_landmarks[i]];

            debug_points3d.push_back(Eigen::Vector3d(lm.point3d.x, lm.point3d.y, lm.point3d.z));
            // debug_points2d1.push_back(lm.view_data[lm.view_data.size() - 2].feature_kp.pt);
            // debug_points2d2.push_back(lm.view_data.back().feature_kp.pt);

            feature_matches.push_back(std::make_pair(lm.view_data[lm.view_data.size() - 2].feature_id, lm.view_data.back().feature_id));
        }

        // DEBUG_visualize_features(cam->rgb, debug_points2d);
        DEBUG_visualize_matches(ref_cam, cam, feature_matches, view_time);

        if (!view_3d)
            return;

        auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(debug_points3d));
        const auto cframe = o3d::geometry::TriangleMesh::CreateCoordinateFrame();
        o3d::visualization::DrawGeometries({ cloud, cframe });
    }

    /**
     *  uses pair-wise homography calculations to filter outliers.
     *  candidate/landmark_matches: <candidate/landmark_id, feature_id>.
     *  If camera_is_new == false, homography calculations will not be made by default
     */
    inline std::vector<std::pair<uint32_t, uint32_t>> cdlm_pairwise_homography_filter(
        const Camera* cam, 
        const std::vector<std::pair<uint32_t, uint32_t>>& cdlm_matches,
        const std::vector<Landmark>& cdlms,
        std::map<std::pair<uint32_t, uint32_t>, cv::Mat>& camera_hgraphy_pairs,
        const bool camera_is_new = false) const
    {
        std::vector<std::pair<uint32_t, uint32_t>> good_matches;
        good_matches.reserve(cdlm_matches.size());
        const uint32_t cam_id = cam->id;

        /**
         *  TODO:
         *      camera_hgraphy_pairs should be cached only for this
         *      frame's run, highly unlikely the same data will be used during later     
         * 
         *  NOTE:
         *      camera_is_new == true:
         *      - collect all unique first -> camera_ids into a set
         *      - dispatch homography calculations for every pair in set
         */

        std::set<uint32_t> invalid_ids;
        std::mutex invalid_mtx;

        std::unordered_map<uint32_t, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>> frame_wise_matches;

        // this is practically free, might as well do it here
        for (const auto match : cdlm_matches)
        {
            const Landmark& cdlm = cdlms[match.first];
            const uint32_t cid = cdlm.view_data.back().camera_id;
            const cv::Point2f fid = cdlm.view_data.back().feature_kp.pt;

            if (frame_wise_matches.find(cid) == frame_wise_matches.end())
            {
                const std::vector<cv::Point2f> x1 = { fid };
                const std::vector<cv::Point2f> x2 = { cam->kp_features[match.second].pt };
                frame_wise_matches.insert(std::make_pair(cid, std::make_pair(x1, x2)));
            }
            else
            {
                frame_wise_matches[cid].first.push_back(fid);
                frame_wise_matches[cid].second.push_back(cam->kp_features[match.second].pt);
            }
        }

        if (camera_is_new)
        {
            Timer t;

            for (const auto fwm : frame_wise_matches)
            {
                // 4 or more would be enough for homography calculations, but 
                // 10 yields more robust results
                if (!(fwm.second.first.size() >= HOMOGRAPHY_MIN_MATCHES))
                {
                    invalid_ids.insert(fwm.first);
                    continue;
                }

                const auto [hgraphy, success] = find_homography_camera_pair_matched(fwm.second);
                if (success)
                    camera_hgraphy_pairs.insert(std::make_pair(std::make_pair(fwm.first, cam_id), hgraphy));
                else
                    invalid_ids.insert(fwm.first);
            }

            #ifdef DEBUG_VERBOSE
            t.stop("homography creation");
            #endif
        }

        // it isn't useful to try to parallelize this, would require too much mutexing
        for (int i = 0; i < cdlm_matches.size(); i++)
        {
            const auto match = cdlm_matches[i];

            const Landmark& cdlm = cdlms.at(match.first);
            const uint32_t ref_cam_id = cdlm.view_data.back().camera_id;

            if (invalid_ids.find(ref_cam_id) != invalid_ids.end())
                continue;

            cv::Mat homography;

            const auto end = camera_hgraphy_pairs.end();
            const auto found = camera_hgraphy_pairs.find(std::make_pair(ref_cam_id, cam_id));

            // hgraphy not yet exists, create
            if (found == end)
            {
                if (!(frame_wise_matches[ref_cam_id].first.size() >= HOMOGRAPHY_MIN_MATCHES))
                {
                    invalid_ids.insert(ref_cam_id);
                    continue;
                }

                const auto [hgraphy, success] = find_homography_camera_pair_matched(frame_wise_matches[ref_cam_id]);
                if (success)
                {
                    camera_hgraphy_pairs.insert(std::make_pair(std::make_pair(ref_cam_id, cam_id), hgraphy));
                    homography = hgraphy;
                }
                else
                {
                    invalid_ids.insert(ref_cam_id);
                    continue;
                }
            }
            else
            {
                homography = found->second;
            }

            // check with hgraphy if outlier

            const cv::Point2f ref_pt = cdlm.view_data.back().feature_kp.pt;
            const cv::Point2f cam_pt = cam->kp_features[match.second].pt;

            cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = ref_pt.x;
            col.at<double>(1) = ref_pt.y;
            col = homography * col;
            col /= col.at<double>(2);

            const double dist = sqrt(pow(col.at<double>(0) - cam_pt.x, 2) + pow(col.at<double>(1) - cam_pt.y, 2));
            if (dist < CANDIDATE_INLIER_THRESHOLD * CANDIDATE_INLIER_THRESHOLD)
            {
                good_matches.emplace_back(match);
            }
        }

        good_matches.shrink_to_fit();
        return good_matches;
    }

    inline std::tuple<cv::Mat, bool> find_homography_camera_pair_matched(const std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& matches) const
    {
        const cv::Mat hgraphy = findHomography(matches.first, matches.second, cv::RANSAC, ransac_thresh, cv::noArray(), FIND_HOMOGRAPHY_RANSAC_ITERATIONS);
        return std::make_tuple(hgraphy, hgraphy.cols != 0);
    }
    
    /**
     *  the hgraphy matrix, succes
     */
    inline std::tuple<cv::Mat, bool> find_homography_camera_pair(const std::pair<uint32_t, uint32_t> cam_pair, const std::vector<Landmark>& cdlms) const
    {
        const Camera* ref_cam = cameras.at(cam_pair.first);
        const Camera* cam = cameras.at(cam_pair.second);

        // std::unique_ptr<cv::BFMatcher> crosscheck_matcher = std::make_unique<cv::BFMatcher>(*cv::BFMatcher::create(cv::NORM_L2, true));

        std::vector<cv::DMatch> nn_matches;
        f_matcher_crosscheck->match(ref_cam->feature_descriptors, cam->feature_descriptors, nn_matches);

        std::vector<cv::Point2f> matched1, matched2;
        matched1.reserve(nn_matches.size());
        matched2.reserve(nn_matches.size());
        
        for (int i = 0; i < nn_matches.size(); i++)
        {
            matched1.emplace_back(ref_cam->kp_features[nn_matches[i].queryIdx].pt);
            matched2.emplace_back(cam->kp_features[nn_matches[i].trainIdx].pt);
        }

        if (matched1.size() >= 4)
        {
            const cv::Mat homography = findHomography(matched1, matched2, cv::RANSAC, ransac_thresh, cv::noArray(), FIND_HOMOGRAPHY_RANSAC_ITERATIONS);
            return std::make_tuple(homography, true);
        }
        else
        {
            // not enough matches
            return std::make_tuple(cv::Mat(), false);
        }
    }

    inline bool candidate_movement_large_enough(const Landmark& cd) const
    {
        /**
         *  the first translation in the pgraph is always 1.0,
         *  use it as reference
         */

        const Camera* first = cameras.at((cd.view_data.front().camera_id));
        const Camera* last = cameras.at((cd.view_data.back().camera_id));
        
        // if translation large enough or angular rotation large enough
        if ((last->position - first->position).norm() > CANDIDATE_TRANSLATION_THRESHOLD ||
            math::rad2deg((math::eigen_rot2euler(last->rotation) - math::eigen_rot2euler(first->rotation)).norm()) > CANDIDATE_ROTATION_THRESHOLD)
        {
            return true;
        }

        return false;
    }

    inline double landmark_average_magnitude() const
    {
        double magnitude_sum = 0.0;

        for (const Landmark& lm : landmarks)
            magnitude_sum += (double)sqrt(lm.point3d.dot(lm.point3d));

        return magnitude_sum / (double)landmarks.size();
    }

    // void candidates_match_against_all();

    void all_candidates_to_landmarks();

    double pointcloud_scale = 0.0;

    std::vector<std::shared_ptr<o3d::geometry::PointCloud>> ptrclouds;

    // <cmaera_id, <candidate feature ids>, contains the unmatched
    // per-camera features
    std::map<uint32_t, std::vector<uint32_t>> camera_candidates;

    double delta_time;

    PosePredictor pose_predictor;

    registration::RegistrationParams reg_params;

    /**
     *  < <camera_id, vector of unmatched feature indexes> >
     */
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> camera_unmatched;
    
};

}