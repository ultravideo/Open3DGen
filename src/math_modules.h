#pragma once
#include <Open3D/Open3D.h>
#include <cmath>
#include <exception>
#include <algorithm>
#include <tuple>
#include <array>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include "constants.h"
#include "s3d_camera.h"


namespace stitcher3d
{
namespace math
{

inline double depth_from_RtX(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Vector3d& X)
{
    return (R*X)(2) + t(2);
}

void triangulate_dlt(const Mat34& P1, const Eigen::Vector2d& x1,
                    const Mat34& P2, const Eigen::Vector2d& x2,
                    Eigen::Vector3d* X_homogeneous);

void triangulate_dlt(const Mat34& P1, const Eigen::Vector2d& x1,
                    const Mat34& P2, const Eigen::Vector2d& x2,
                    Eigen::Vector4d *X_euclidean);

void homogeneous_to_euclidean(const Eigen::Vector4d& H, Eigen::Vector3d* X);

Eigen::Vector4d euclidean_to_homogenous(const Eigen::Vector3d eucl);

void P_From_KRt(const Eigen::Matrix3d& intr, const Eigen::Matrix3d& rotation, 
    const Eigen::Vector3d& translation, Mat34 *P);

inline Mat34 P_From_KT(const Eigen::Matrix3d& intr, const Eigen::Matrix4d T)
{
    Mat34 P = Mat34::Identity();

    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<1, 3>(3, 0);

    P.block<3, 3>(0, 0) = R.transpose();
    P.col(3) = -R.transpose() * t;

    return intr * P;
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> solution_in_front_of_camera(const std::vector<Eigen::Matrix3d>& rotations, 
    const std::vector<Eigen::Vector3d>& translations, const Eigen::Matrix3d& intr1, const Eigen::Matrix3d& intr2,
    const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);

template <typename TMat, typename TVec>
double compute_nullspace(TMat *A, TVec *nullspace);

std::tuple<Eigen::Matrix3d, double> compute_essential_8_point(const std::vector<Eigen::Vector2d>& x1, 
    const std::vector<Eigen::Vector2d>& x2);

std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>> 
    r_and_t_from_essential_matrix(const Eigen::Matrix3d& E);

Eigen::Quaterniond quaternion_look_rotation(const Eigen::Vector3d& forward, 
            const Eigen::Vector3d& up = UP_VECTOR);


std::tuple<Eigen::Vector3d, bool> intersect_trianle_ray(const Eigen::Vector3d& origin,
                const Eigen::Vector3d& direction,
                const Eigen::Vector3i& triangle, const std::vector<Eigen::Vector3d>& vertices,
                const Eigen::Vector3d& tr_normal);

std::tuple<Eigen::Vector3d, bool, int> raycast_from_point_to_surface(const Eigen::Vector3d& origin,
                const Eigen::Vector3d& direction,
                const std::vector<Eigen::Vector3d>& vertices,
                const std::vector<Eigen::Vector3i>& triangles,
                const std::vector<Eigen::Vector3d>& triangle_normals);

inline float vector_angle(const Eigen::Vector2d& f, const Eigen::Vector2d& s)
{
    return acos((f.normalized()).dot(s.normalized()));
}

inline float vector_angle_ccw(const Eigen::Vector2d& f, const Eigen::Vector2d& s)
{
    return atan2(f.x() * s.y() - f.y() * s.x(), f.x() * s.x() + f.y() * s.y());
}

inline Eigen::Matrix2d get_2d_rotation_matrix(const float angle)
{
    Eigen::Matrix2d ret;
    ret << cos(angle), -sin(angle), sin(angle), cos(angle);
    return ret;
}

inline Eigen::Vector3d pixel_to_world(const Eigen::Vector2d& pixel, 
    const float d_val, const o3d::camera::PinholeCameraIntrinsic& intr)
{
    auto [cx, cy] = intr.GetPrincipalPoint();
    auto [fx, fy] = intr.GetFocalLength();

    const Eigen::Vector4d world_point(
        ((float)pixel[0] - cx) * d_val / fx,
        ((float)pixel[1] - cy) * d_val / fy,
        d_val,
        1.0
    );

    const auto flipped_point = FLIP_TRANSFORM_4D * world_point;

    return Eigen::Vector3d(flipped_point[0], flipped_point[1], flipped_point[2]);
}

inline bool is_close(float f, float s)
{
    return abs(f - s) < FLOAT_SMALL;
}

inline double rad2deg(const double rad)
{
    return rad * 180.0 / 3.141592653589793238463;
}

inline double deg2rad(const double deg)
{
    return deg * 3.141592653589793238463 / 180.0;
}

/**
 *  Converts a given Rotation Matrix to Euler angles
 *  Convention used is Y-Z-X Tait-Bryan angles
 *  Reference code implementation:
 *  https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
 *  
 *  taken from https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/Utils.cpp
 */
inline cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
    cv::Mat euler(3,1,CV_64F);

    double m00 = rotationMatrix.at<double>(0,0);
    double m02 = rotationMatrix.at<double>(0,2);
    double m10 = rotationMatrix.at<double>(1,0);
    double m11 = rotationMatrix.at<double>(1,1);
    double m12 = rotationMatrix.at<double>(1,2);
    double m20 = rotationMatrix.at<double>(2,0);
    double m22 = rotationMatrix.at<double>(2,2);

    double bank, attitude, heading;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        bank = 0;
        attitude = CV_PI/2;
        heading = atan2(m02,m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        bank = 0;
        attitude = -CV_PI/2;
        heading = atan2(m02,m22);
    }
    else
    {
        bank = atan2(-m12,m11);
        attitude = asin(m10);
        heading = atan2(-m20,m00);
    }

    euler.at<double>(0) = bank;
    euler.at<double>(1) = attitude;
    euler.at<double>(2) = heading;

    return euler;
}

inline Eigen::Vector3d eigen_rot2euler(const Eigen::Matrix3d rotation)
{
    cv::Mat ref_rot_cv;
    cv::eigen2cv(rotation, ref_rot_cv);
    const cv::Mat euler_cv = rot2euler(ref_rot_cv);
    const Eigen::Vector3d euler (euler_cv.at<double>(0), euler_cv.at<double>(1), euler_cv.at<double>(2));

    return euler;
}

/**
 * Converts a given Euler angles to Rotation Matrix
 * Convention used is Y-Z-X Tait-Bryan angles
 * Reference:
 * https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
 * 
 * taken from https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/Utils.cpp
 */
inline cv::Mat euler2rot(const cv::Mat & euler)
{
    cv::Mat rotationMatrix(3,3,CV_64F);

    double bank = euler.at<double>(0);
    double attitude = euler.at<double>(1);
    double heading = euler.at<double>(2);

    // Assuming the angles are in radians.
    double ch = cos(heading);
    double sh = sin(heading);
    double ca = cos(attitude);
    double sa = sin(attitude);
    double cb = cos(bank);
    double sb = sin(bank);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh*sb - ch*sa*cb;
    m02 = ch*sa*sb + sh*cb;
    m10 = sa;
    m11 = ca*cb;
    m12 = -ca*sb;
    m20 = -sh*ca;
    m21 = sh*sa*cb + ch*sb;
    m22 = -sh*sa*sb + ch*cb;

    rotationMatrix.at<double>(0,0) = m00;
    rotationMatrix.at<double>(0,1) = m01;
    rotationMatrix.at<double>(0,2) = m02;
    rotationMatrix.at<double>(1,0) = m10;
    rotationMatrix.at<double>(1,1) = m11;
    rotationMatrix.at<double>(1,2) = m12;
    rotationMatrix.at<double>(2,0) = m20;
    rotationMatrix.at<double>(2,1) = m21;
    rotationMatrix.at<double>(2,2) = m22;

    return rotationMatrix;
}

inline Eigen::Matrix3d eigen_euler2rot(const Eigen::Vector3d euler)
{
    cv::Mat euler_cv (3, 1, CV_64F);
    cv::eigen2cv(euler, euler_cv);
    const cv::Mat rot_cv = euler2rot(euler_cv);

    Eigen::Matrix3d rot;
    cv::cv2eigen(rot_cv, rot);

    return rot;
}

inline std::vector<Eigen::Vector2d> pixel_space_to_image_space(const std::vector<Eigen::Vector2d>& points, const o3d::camera::PinholeCameraIntrinsic& intr)
{
    std::vector<Eigen::Vector2d> img_points;
    const auto [fx, fy] = intr.GetFocalLength();
    const auto [cx, cy] = intr.GetPrincipalPoint();
    const int width = intr.width_;
    const int height = intr.height_;

    for (const auto& p : points)
    {
        Eigen::Vector3d temp (p.x(), p.y(), 1.0);
        temp = intr.intrinsic_matrix_.inverse() * temp;
        img_points.emplace_back(Eigen::Vector2d(temp.x(), temp.y()));
        // img_points.emplace_back(Eigen::Vector2d(
            // fx * (p.x() / (double)(width)) + cx,
            // fy * (p.y() / (double)(height)) + cy
            // p.x() / (double)(width),
            // p.y() / (double)(width)
            // (p.x() - cx) / fx,
            // (p.y() - cy) / fy
        // ));
    }

    // std::cout << points[0] << "\n" << img_points[0] << "\n\n";
    
    return img_points;
}


template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline bool point_infront_of_camera(const Mat34 P, const Eigen::Vector3d point)
{
    const Eigen::Vector4d h = euclidean_to_homogenous(point);

    double condition_1 = P.row(2).dot(h) * h[3];
    double condition_2 = h[2] * h[3];

    if (condition_1 > 0 && condition_2 > 0)
        return true;
    else
        return false;
}

}
}
