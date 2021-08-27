#pragma once

#include <Open3D/Open3D.h>
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "constants.h"
#include "utilities.h"
#include "timer.h"


namespace stitcher3d
{
namespace rgbd
{
/**
 * @brief creates a pointcloud from a given rgb-d -pair
 */
std::shared_ptr<o3d::geometry::PointCloud> create_pcloud_from_rgbd(
                    o3d::geometry::Image& rgb, o3d::geometry::Image& depth,
                    float depth_scale, float depth_clip_dist, bool invert,
                    const o3d::camera::PinholeCameraIntrinsic& intr);


std::tuple<std::shared_ptr<o3d::geometry::PointCloud>, o3d::geometry::Image, o3d::geometry::Image>
    get_pointcloud_and_images(const std::string& rgb_filepath,
    const std::string& d_filepath,
    const o3d::camera::PinholeCameraIntrinsic& intr,
    bool invert = true);

inline Eigen::Vector3d triangulate_point(const double u, const double v, const double d, const o3d::camera::PinholeCameraIntrinsic& intr, const bool invert = true)
{
    const auto [cx, cy] = intr.GetPrincipalPoint();
    const auto [fx, fy] = intr.GetFocalLength();

    const double x = (u - cx) * d / fx;
    const double y = (v - cy) * d / fy;

    if (!invert)
        return Eigen::Vector3d(x, y, d);
    
    return FLIP_TRANSFORM_3D * Eigen::Vector3d(x, y, d);;
}

inline Eigen::Vector3d triangulate_point(const Eigen::Vector2d& uv, const double d, const o3d::camera::PinholeCameraIntrinsic& intr, const bool invert = true)
{
    return triangulate_point(uv.x(), uv.y(), d, intr, invert);
}

enum CV_IMG_TYPE
{
    TYPE_CV_8UC3 = 1,
    TYPE_CV_16S = 2
};

inline o3d::geometry::Image cv2o3d(const cv::Mat img, const CV_IMG_TYPE type_format)
{
    /**
     *  TODO: parallelize this
     */

    o3d::geometry::Image o3i;
    o3i.Prepare(img.cols, img.rows, img.channels(), type_format);

    int channels;
    if (type_format == CV_IMG_TYPE::TYPE_CV_16S)
        channels = 1;
    else if (type_format == CV_IMG_TYPE::TYPE_CV_8UC3)
        channels = 3;

    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            for (int c = 0; c < channels; c++)
            {
                if (type_format == CV_IMG_TYPE::TYPE_CV_8UC3)
                {
                    const uint8_t p = img.at<cv::Vec3b>(y, x)[c];
                    *o3i.PointerAt<uint8_t>(x, y, c) = p;
                }
                else if (type_format == CV_IMG_TYPE::TYPE_CV_16S)
                {
                    const uint16_t p = img.at<uint16_t>(y, x);
                    *o3i.PointerAt<uint16_t>(x, y) = p;
                }
            }
        }
    }

    return o3i;
}

inline float hue_to_rgb(float v1, float v2, float vH)
{
    if (vH < 0.0)
        vH += 1.0;

    if (vH > 1)
        vH -= 1;

    if ((6 * vH) < 1)
        return (v1 + (v2 - v1) * 6 * vH);

    if ((2 * vH) < 1)
        return v2;

    if ((3 * vH) < 2)
        return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

    return v1;
}

inline std::tuple<float, float, float> HSL_to_RGB(const float h, const float s, const float l)
{
    float r, g, b;

    if (s == 0)
    {
        r = g = b = (l * 255.0);
    }
    else
    {
        float v1, v2;
        float hue = (float)h / 360.0;

        v2 = (l < 0.5) ? (l * (1.0 + s)) : ((l + s) - (l * s));
        v1 = 2.0 * l- v2;

        r = (255.0 * hue_to_rgb(v1, v2, hue + (1.0f / 3.0)));
        g = (255.0 * hue_to_rgb(v1, v2, hue));
        b = (255.0 * hue_to_rgb(v1, v2, hue - (1.0f / 3.0)));
    }

    return std::make_tuple(r, g, b);
}

inline cv::Mat o3d_image_to_cv_image(const std::shared_ptr<o3d::geometry::Image> o3d_img, const bool is_depth = false)
{
    cv::Mat img (o3d_img->height_, o3d_img->width_, CV_8UC3);

    for (int y = 0; y < o3d_img->height_; y++)
    {
        for (int x = 0; x < o3d_img->width_; x++)
        {
            if (is_depth)
            {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    *o3d_img->PointerAt<uint16_t>(x, y, 0) / 2,
                    *o3d_img->PointerAt<uint16_t>(x, y, 1) / 2,
                    *o3d_img->PointerAt<uint16_t>(x, y, 2)) / 2;
            }
            else
            {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    *o3d_img->PointerAt<uint8_t>(x, y, 0),
                    *o3d_img->PointerAt<uint8_t>(x, y, 1),
                    *o3d_img->PointerAt<uint8_t>(x, y, 2));
            }
        }
    }

    // cv::imshow("output", img);
    // cv::waitKey(0);

    return img;
}

}
}