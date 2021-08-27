#pragma once

#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <Open3D/Open3D.h>
#include <string>
#include <tuple>
#include <vector>
#include <fstream>
#include <memory>
#include <streambuf>
#include <exception>
#include <iostream>
#include <stdio.h>
#include "constants.h"
#include "utilities.h"
#include "timer.h"

namespace stitcher3d
{
namespace shader
{

static bool compute_shaders_init = false;

struct pixel_data
{
    float p_x;
    float p_y;
    float p_z;

    int tr_i;

    int32_t coord_x;
    int32_t coord_y;

    float padding0;
    float padding1;
};

struct vertex_data
{
    float v_x;
    float v_y;
    float v_z;
    float padding;
};

struct shader_info
{
    GLuint shader_program;
    GLuint compute_shader;
    unsigned int img_width;
    unsigned int img_height;
    
    int hit_point_buffer_offset;
    int triangle_buffer_offset;
    
    GLuint triangle_ssbo;
    GLuint depth_texture_id;
    GLuint hit_point_ssbo;
    GLuint intr_matrix_offset;

    o3d::camera::PinholeCameraIntrinsic intr_matrix;

    int triangle_buffer_size;

    // float* hit_point_buffer;
    std::shared_ptr<std::vector<float>> hit_point_buffer;;
    std::shared_ptr<std::vector<vertex_data>> triangle_buffer;
    std::shared_ptr<std::vector<float>> depth_buffer;

    std::vector<float> camera_transform;

    GLFWwindow* window;

    float depth_cloud_world_scale;

    uint32_t workgroup_size;
};

bool get_gl_error(GLuint shader_i);

const std::string read_shader(const std::string& shader_path);

GLuint compile_shader(const std::string& shader);

void setup_compute_shaders(shader_info& s_info);

std::shared_ptr<std::vector<pixel_data>> dispatch_compute_shader(shader_info& s_info);

void create_compute_shader();

void shader_cleanup(shader_info s_info);

}
}