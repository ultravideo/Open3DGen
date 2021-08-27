#pragma once

#include <Open3D/Open3D.h>
#include <cstdint>
#include <memory>
#include <array>
#include <exception>
#include <vector>
#include <algorithm>
#include "timer.h"
#include "constants.h"
#include "surface_mesh.h"
#include "math_modules.h"
#include "uv_unwrapper.h"
#include "shader_stuff.h"

namespace stitcher3d
{
namespace textures
{

/**
 * dilates the texture, tries to mitigate uv seem bleeding
 */
void dilate_image(o3d::geometry::Image& to_filter);

std::shared_ptr<o3d::geometry::Image> filter_image_3ch(const o3d::geometry::Image& to_filter, o3d::geometry::Image::FilterType ftype);

void project_texture_on_mesh_gpu(const o3d::geometry::Image& rgb, const o3d::geometry::Image d, 
        const o3d::camera::PinholeCameraIntrinsic& camera_intr, const float depth_scale, const float depth_cloud_world_scale,
        surface::SurfaceMesh& mesh, o3d::geometry::Image& texture,
        const Eigen::Matrix4d_u camera_Transform,
        shader::shader_info& s_info,
        const float ray_threshold);

void project_texture_on_mesh(o3d::geometry::Image& rgb, o3d::geometry::Image d, 
        const o3d::camera::PinholeCameraIntrinsic& camera_intr, const float depth_scale,
        surface::SurfaceMesh& mesh, o3d::geometry::Image& texture);

void project_texture_sequence_on_mesh_gpu(const float depth_scale, const float depth_cloud_world_scale, surface::SurfaceMesh& mesh, std::vector<o3d::geometry::Image>& textures,
        const std::vector<std::shared_ptr<Camera>> cameras, const uint32_t project_every_nth, const uint32_t workgroup_size, const float ray_threshold, const float reject_blur);

void non_zero_convolve_uv_filter(o3d::geometry::Image& texture);

o3d::geometry::Image average_blend_images(const std::vector<o3d::geometry::Image>& textures);

o3d::geometry::Image non_overwrite_blend_images(const std::vector<o3d::geometry::Image>& textures);


static const std::array<Eigen::Vector2i, 8> non_zero_index_offsets {
        Eigen::Vector2i(0, -1),
        Eigen::Vector2i(1, -1),
        Eigen::Vector2i(1, 0),
        Eigen::Vector2i(1, 1),
        Eigen::Vector2i(0, 1),
        Eigen::Vector2i(-1, 1),
        Eigen::Vector2i(-1, 0),
        Eigen::Vector2i(-1, -1)
};

}
}
