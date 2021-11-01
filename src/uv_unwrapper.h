#pragma once

#include <Open3D/Open3D.h>
#include <cstdint>
#include <memory>
#include <tuple>
#include <algorithm>
#include <vector>
#include "constants.h"
#include "surface_mesh.h"
#include "math_modules.h"
#include "../libraries/tinyobj/tiny_obj_loader.h"
// #include "../Open3D/3rdparty/tinyobjloader/tiny_obj_loader.h"
#include "../libraries/xatlas/xatlas.h"


namespace stitcher3d
{
namespace uv
{
  
template <typename T>
inline T min_arg(const T &a, const T &b)
{
  return a < b ? a : b;
}

void unwrap_uvs_xatlas(std::shared_ptr<surface::SurfaceMesh> mesh);

Eigen::Vector2d get_point_uv(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& p2, const Eigen::Vector3d& p,
                            const Eigen::Vector2d& uv0, const Eigen::Vector2d& uv1,
                            const Eigen::Vector2d& uv2);

void calculate_individual_uvs(std::shared_ptr<surface::SurfaceMesh> mesh, 
                            const float uv_margin);

std::vector<unsigned int> sort_uv_triangles_by_area(
            const std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* uvs);


inline void lay_uv_flat(std::array<Eigen::Vector2d, TR_VERT_COUNT>& uv)
{
    const std::array<float, TR_VERT_COUNT> angles {
        math::vector_angle(uv[1] - uv[0], uv[2] - uv[0]),
        math::vector_angle(uv[0] - uv[1], uv[2] - uv[1]),
        math::vector_angle(uv[1] - uv[2], uv[0] - uv[2])
    };

    int max_angle_i = std::distance(angles.begin(), std::max_element(angles.begin(), angles.end()));
    Eigen::Vector2d longest_side;
    if (max_angle_i == 0)
        longest_side = uv[1] - uv[2];
    else if (max_angle_i == 1)
        longest_side = uv[0] - uv[2];
    else if (max_angle_i == 2)
        longest_side = uv[0] - uv[1];

    const float longest_side_angle = math::vector_angle_ccw(longest_side, RIGHT_UV);

    const Eigen::Matrix2d rotation_matrix = math::get_2d_rotation_matrix(longest_side_angle);

    for (int i = 0; i < uv.size(); i++)
        uv[i] = rotation_matrix * uv[i];
}

void pad_uvs(std::vector<Eigen::Vector2d>& uv_coords, const std::vector<uint32_t>& indices, const double pad_scale);

inline std::vector<double> flatten_vec_uvs(const std::vector<Eigen::Vector2d>& uvs)
{
    std::vector<double> flattened;
    flattened.reserve(uvs.size() * 2);

    for (const auto& uv : uvs)
    {
        flattened.push_back(uv.x());
        flattened.push_back(uv.y());
    }

    return flattened;
}

inline std::tuple<float, float, float, float> get_uv_bounds(
                    const std::array<Eigen::Vector2d, TR_VERT_COUNT>& uv)
{
    const std::array<float, TR_VERT_COUNT> x_coords {
        (float)uv[0].x(),
        (float)uv[1].x(),
        (float)uv[2].x()
    };

    const std::array<float, TR_VERT_COUNT> y_coords {
        (float)uv[0].y(),
        (float)uv[1].y(),
        (float)uv[2].y()
    };

    return std::make_tuple(
        *std::min_element(x_coords.begin(), x_coords.end()),
        *std::max_element(x_coords.begin(), x_coords.end()),
        *std::min_element(y_coords.begin(), y_coords.end()),
        *std::max_element(y_coords.begin(), y_coords.end())
    );
}

inline void pack_uv(std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* uvs, 
                    const float uv_margin, const int index, float& current_row_height,
                    float& previous_row_height, const std::vector<unsigned int>& idx)
{
    std::array<Eigen::Vector2d, TR_VERT_COUNT> uv = uvs->at(idx[index]);

    const int prev_i = index - 1;
    lay_uv_flat(uv);
    const auto [xmin, xmax, ymin, ymax] = get_uv_bounds(uv);

    if (prev_i < 0)
    {
        for (Eigen::Vector2d& v : uv)
        {
            v[0] = v[0] - xmin + UV_PADDING;
            v[1] = v[1] - ymin + UV_PADDING;
        }

        previous_row_height = UV_PADDING;

        uvs->at(idx[index]) = uv;
        return;
    }

    auto [s_xmin, s_xmax, x_ymin, x_ymax] = get_uv_bounds(uvs->at(idx[prev_i]));
    const float new_max_x = xmax - xmin + s_xmax + UV_PADDING;

    if (new_max_x >= UV_MAX_X)
    {
        s_xmax = 0.f;
        previous_row_height = current_row_height + UV_PADDING;
    }

    for (Eigen::Vector2d& v : uv)
    {
        v[0] = v[0] - xmin + s_xmax + UV_PADDING;
        v[1] = v[1] - ymin + previous_row_height;

        current_row_height = std::max(current_row_height, (float)v[1]);
    }

    uvs->at(idx[index]) = uv;
}

inline void pack_uvs(std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* uvs, 
                    const float uv_margin, const std::vector<unsigned int>& idx)
{
    float current_row_height = 0;
    float previous_row_height = 0;
    for (int i = 0; i < uvs->size(); i++)
        pack_uv(uvs, uv_margin, i, current_row_height, previous_row_height, idx);
}



}
}
