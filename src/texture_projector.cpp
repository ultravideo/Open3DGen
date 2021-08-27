#include "texture_projector.h"
#include "rgbd_utilities.h"
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace stitcher3d
{
namespace textures
{

void dilate_image(o3d::geometry::Image& to_filter)
{
    for (int x = 0; x < to_filter.width_; x++)
    {
        for (int y = 0; y < to_filter.width_; y++)
        {
            uint8_t* rpixel = to_filter.PointerAt<uint8_t>(x, y, 0);
            uint8_t* gpixel = to_filter.PointerAt<uint8_t>(x, y, 1);
            uint8_t* bpixel = to_filter.PointerAt<uint8_t>(x, y, 2);

            // if pixel is zero
            if (*rpixel == 0 && *gpixel == 0 && *bpixel == 0)
            {
                // take the average of pixels around it
                uint32_t ravg = 0;
                uint32_t bavg = 0;
                uint32_t gavg = 0;
                uint32_t count = 0;

                for (const Eigen::Vector2i& offset : non_zero_index_offsets)
                {
                    if (x + offset[0] < 0 || x + offset[0] >= to_filter.width_ ||
                        y + offset[1] < 0 || y + offset[1] >= to_filter.height_)
                        continue;

                    uint8_t* rr = to_filter.PointerAt<uint8_t>(x + offset[0], y + offset[1], 0);
                    uint8_t* gg = to_filter.PointerAt<uint8_t>(x + offset[0], y + offset[1], 1);
                    uint8_t* bb = to_filter.PointerAt<uint8_t>(x + offset[0], y + offset[1], 2);

                    if (*rr == 0 && *gg == 0 && *bb == 0)
                        continue;

                    ravg += *rr;
                    gavg += *gg;
                    bavg += *bb;
                    count++;
                }

                *rpixel = (uint8_t)(float(ravg) / float(count));
                *gpixel = (uint8_t)(float(gavg) / float(count));
                *bpixel = (uint8_t)(float(bavg) / float(count));
            }
        }
    }
}

std::shared_ptr<o3d::geometry::Image> filter_image_3ch(const o3d::geometry::Image& to_filter, o3d::geometry::Image::FilterType ftype)
{
    o3d::geometry::Image red_channel;
    o3d::geometry::Image green_channel;
    o3d::geometry::Image blue_channel;
    red_channel.Prepare(to_filter.width_, to_filter.height_, 1, to_filter.bytes_per_channel_);
    green_channel.Prepare(to_filter.width_, to_filter.height_, 1, to_filter.bytes_per_channel_);
    blue_channel.Prepare(to_filter.width_, to_filter.height_, 1, to_filter.bytes_per_channel_);

    for (int x = 0; x < to_filter.width_; x++)
    {
        for (int y = 0; y < to_filter.height_; y++)
        {
            *red_channel.PointerAt<uint8_t>(x, y, 0) = *to_filter.PointerAt<uint8_t>(x, y, 0);
            *green_channel.PointerAt<uint8_t>(x, y, 0) = *to_filter.PointerAt<uint8_t>(x, y, 1);
            *blue_channel.PointerAt<uint8_t>(x, y, 0) = *to_filter.PointerAt<uint8_t>(x, y, 2);
        }
    }

    auto red_ch_float = red_channel.CreateFloatImage(o3d::geometry::Image::ColorToIntensityConversionType::Weighted);
    auto green_ch_float = green_channel.CreateFloatImage(o3d::geometry::Image::ColorToIntensityConversionType::Weighted);
    auto blue_ch_float = blue_channel.CreateFloatImage(o3d::geometry::Image::ColorToIntensityConversionType::Weighted);

    auto red_channel_blurred = red_ch_float->Filter(ftype);
    auto green_channel_blurred = green_ch_float->Filter(ftype);
    auto blue_channel_blurred = blue_ch_float->Filter(ftype);

    std::shared_ptr<o3d::geometry::Image> blurred_3ch = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
    blurred_3ch->Prepare(to_filter.width_, to_filter.height_, 3, to_filter.bytes_per_channel_);

    for (int x = 0; x < to_filter.width_; x++)
    {
        for (int y = 0; y < to_filter.height_; y++)
        {
            *blurred_3ch->PointerAt<uint8_t>(x, y, 0) = (uint8_t)(*red_channel_blurred->PointerAt<float>(x, y, 0) * 255.f);
            *blurred_3ch->PointerAt<uint8_t>(x, y, 1) = (uint8_t)(*green_channel_blurred->PointerAt<float>(x, y, 0) * 255.f);
            *blurred_3ch->PointerAt<uint8_t>(x, y, 2) = (uint8_t)(*blue_channel_blurred->PointerAt<float>(x, y, 0) * 255.f);
        }
    }

    return blurred_3ch;
}

void non_zero_convolve_uv_filter(o3d::geometry::Image& texture)
{
    std::shared_ptr<o3d::geometry::Image> blurred = filter_image_3ch(texture, o3d::geometry::Image::FilterType::Gaussian5);

    for (int x = 0; x < texture.width_; x++)
    {
        for (int y = 0; y < texture.height_; y++)
        {
            const auto sample_to_r = texture.PointerAt<uint8_t>(x, y, 0);
            const auto sample_to_g = texture.PointerAt<uint8_t>(x, y, 1);
            const auto sample_to_b = texture.PointerAt<uint8_t>(x, y, 2);

            // if (*texture.PointerAt<uint8_t>(x, y, 0) > BLACK_THRESHOLD || *texture.PointerAt<uint8_t>(x, y, 1) > BLACK_THRESHOLD || *texture.PointerAt<uint8_t>(x, y, 2) > BLACK_THRESHOLD)
            //     continue;
            if (*sample_to_r > BLACK_THRESHOLD || *sample_to_r > BLACK_THRESHOLD || *sample_to_r > BLACK_THRESHOLD)
                continue;

            const uint8_t red = *(blurred->PointerAt<uint8_t>(x, y, 0));
            const uint8_t green = *(blurred->PointerAt<uint8_t>(x, y, 1));
            const uint8_t blue = *(blurred->PointerAt<uint8_t>(x, y, 2));

            *sample_to_r = red;
            *sample_to_g = green;
            *sample_to_b = blue;
        }
    }

   /*
   int non_z_filtered = 0;
    for (int x = 1; x < texture.width_ - 1; x++)
    {
        for (int y = 1; y < texture.height_ - 1; y++)
        {
            if (*texture.PointerAt<uint8_t>(x, y, 0) != 0 || *texture.PointerAt<uint8_t>(x, y, 1) != 0 || *texture.PointerAt<uint8_t>(x, y, 2) != 0)
                continue;

            int new_red = 0;
            int new_green = 0;
            int new_blue = 0;

            int new_p_count = 0;

            for (int i = 0; i < non_zero_index_offsets.size(); i++)
            {
                const int x_offset = non_zero_index_offsets[i][0];
                const int y_offset = non_zero_index_offsets[i][1];

                uint8_t* up_p_r = texture.PointerAt<uint8_t>(x + x_offset, y + y_offset, 0);
                uint8_t* up_p_g = texture.PointerAt<uint8_t>(x + x_offset, y + y_offset, 1);
                uint8_t* up_p_b = texture.PointerAt<uint8_t>(x + x_offset, y + y_offset, 2);

                if (*up_p_r == 0 && *up_p_g == 0 && *up_p_b == 0)
                    continue;

                new_p_count++;
                new_red += (int)(*up_p_r);
                new_green += (int)(*up_p_g);
                new_blue += (int)(*up_p_b);
            }

            if (new_p_count == 0)
                continue;

            *texture.PointerAt<uint8_t>(x, y, 0) = (uint8_t)(new_red / new_p_count);
            *texture.PointerAt<uint8_t>(x, y, 1) = (uint8_t)(new_green / new_p_count);
            *texture.PointerAt<uint8_t>(x, y, 2) = (uint8_t)(new_blue / new_p_count);
            
            if (*texture.PointerAt<uint8_t>(x, y, 0) != 0 || *texture.PointerAt<uint8_t>(x, y, 1) != 0 || *texture.PointerAt<uint8_t>(x, y, 2) != 0)
                non_z_filtered++;
        }
    }

    std::cout << non_z_filtered << " filrtered\n";
    // o3d::io::WriteImage(DATAFILE_PATH + "texture_out_gpu.png", texture);
    */
}

void project_texture_sequence_on_mesh_gpu(const float depth_scale, const float depth_cloud_world_scale, surface::SurfaceMesh& mesh, std::vector<o3d::geometry::Image>& textures,
        const std::vector<std::shared_ptr<Camera>> cameras, const uint32_t project_every_nth, const uint32_t workgroup_size, const float ray_threshold, const float reject_blur)
{
    if (textures.size() != cameras.size())
        throw std::runtime_error("textures and cameras size didn't match, aborting!");
        
    shader::shader_info s_info;
    s_info.workgroup_size = workgroup_size;
    s_info.intr_matrix = cameras[0]->intr;
    s_info.img_width = cameras[0]->rgb->width_;
    s_info.img_height = cameras[0]->rgb->height_;
    shader::setup_compute_shaders(s_info);

    std::vector<float> frame_sharpness;

    // compute the average sharpness of the images
    float avg_sharpness = 0.f;
    if (reject_blur > 0.f)
    {
        for (int ii = 0; ii < cameras.size(); ii++)
        {
            cv::Mat mblur_laplacian, mblur_mean, mblur_variance;
            cv::Mat cv_img = rgbd::o3d_image_to_cv_image(cameras.at(ii)->rgb);
            cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2GRAY);
            cv::Laplacian(cv_img, mblur_laplacian, CV_64F, 3);

            cv::meanStdDev(mblur_laplacian, mblur_mean, mblur_variance);
            const float sharpness = mblur_variance.at<double>(0) * mblur_variance.at<double>(0);
            avg_sharpness += sharpness;
            frame_sharpness.push_back(sharpness);
        }
        avg_sharpness /= (float)cameras.size();
        std::cout << "the frames have an average sharpness of " << avg_sharpness << "\n";
    }

    std::cout << "begin projecting textures\n";

    for (int ii = 0; ii < cameras.size(); ii++)
    {
        if (ii % project_every_nth != 0)
            continue;

        float bvar = 0.0f;

        // reject frame if it is blurry by comparing it to the average sharpness
        if (reject_blur > 0.f)
        {
            bvar = frame_sharpness[ii];

            if (bvar < reject_blur * avg_sharpness)
            {
                std::cout << "frame " << ii << " out of " << cameras.size() << " had blur varianceo of " 
                    << bvar << " compared to threshold " << reject_blur * avg_sharpness << ", frame is rejected\n";

                continue;
            }
        }

        std::cout << "frame " << ii << " out of " << cameras.size() << " has sharpness of " << bvar << "\n";
        
        project_texture_on_mesh_gpu(*cameras.at(ii)->rgb, *cameras.at(ii)->depth, cameras.at(ii)->intr, depth_scale, depth_cloud_world_scale, mesh, textures[ii], cameras.at(ii)->T, s_info, ray_threshold);
        textures[ii] = *textures[ii].FlipVertical();
    }

    shader::shader_cleanup(s_info);
}

o3d::geometry::Image average_blend_images(const std::vector<o3d::geometry::Image>& textures)
{
    o3d::geometry::Image avg_texture = textures.at(0);

    for (int y = 0; y < avg_texture.height_; y++)
    {
        for (int x = 0; x < avg_texture.width_; x++)
        {
            int avg_r_val = 0;
            int avg_g_val = 0;
            int avg_b_val = 0;
            int blend_count = 0;

            for (int i = 0; i < textures.size(); i++)
            {
                const uint8_t temp_avg_r_val = *textures[i].PointerAt<uint8_t>(x, y, 0);
                const uint8_t temp_avg_g_val = *textures[i].PointerAt<uint8_t>(x, y, 1);
                const uint8_t temp_avg_b_val = *textures[i].PointerAt<uint8_t>(x, y, 2);

                if (temp_avg_b_val == 0 && temp_avg_g_val == 0 && temp_avg_r_val == 0)
                    continue;

                blend_count++;
                avg_r_val += (int)temp_avg_r_val;
                avg_g_val += (int)temp_avg_g_val;
                avg_b_val += (int)temp_avg_b_val;

            }

            if (blend_count != 0)
            {
                avg_r_val = avg_r_val / blend_count;
                avg_g_val = avg_g_val / blend_count;
                avg_b_val = avg_b_val / blend_count;
            }

            *avg_texture.PointerAt<uint8_t>(x, y, 0) = (uint8_t)avg_r_val;
            *avg_texture.PointerAt<uint8_t>(x, y, 1) = (uint8_t)avg_g_val;
            *avg_texture.PointerAt<uint8_t>(x, y, 2) = (uint8_t)avg_b_val;
        }
    }

    return avg_texture;
}

void project_texture_on_mesh_gpu(const o3d::geometry::Image& rgb, const o3d::geometry::Image d, 
        const o3d::camera::PinholeCameraIntrinsic& camera_intr, const float depth_scale, const float depth_cloud_world_scale,
        surface::SurfaceMesh& mesh, o3d::geometry::Image& texture,
        const Eigen::Matrix4d_u camera_transform,
        shader::shader_info& s_info, const float ray_threshold)
{
    const std::vector<Eigen::Vector3d>* vertices = mesh.get_vertices();
    const std::vector<Eigen::Vector3i>* triangles = mesh.get_triangles();
    const std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* triangle_uvs = mesh.get_triangle_uvs();

    // compute shader buffer fillings
    {
        s_info.img_width = rgb.width_;
        s_info.img_height = rgb.height_;
        s_info.hit_point_buffer_offset = GL_HITPOINT_OFFSET;
        s_info.triangle_buffer_offset = GL_TRIANGLE_INDEX_SSBO_OFFSET;
        s_info.triangle_buffer_size = triangles->size() * 3;
        s_info.depth_cloud_world_scale = depth_cloud_world_scale;

        std::cout << s_info.img_width << ", " << s_info.img_height << "\n\n";
        
        // convert the transform matrix from OpenCV's format to ogl
        Eigen::Matrix4d_u cv2gl = Eigen::Matrix4d_u::Zero();
        cv2gl.coeffRef(0, 0) = 1.0;
        cv2gl.coeffRef(1, 1) = -1.0;
        cv2gl.coeffRef(2, 2) = -1.0;
        cv2gl.coeffRef(3, 3) = 1.0;
        const Eigen::Matrix4d_u camera_T_major_flip = (cv2gl * camera_transform.transpose()).transpose();

        std::vector<float> camera_transform_float(16);
        for (int i = 0; i < 16; i++)
        {
            camera_transform_float.at(i) = camera_T_major_flip.coeff(i);
        }

        s_info.camera_transform = camera_transform_float;


        auto d_float = d.ConvertDepthToFloatImage();
        // float* float_d_buffer = new float[1280*720];
        std::shared_ptr<std::vector<float>> float_d_buffer = 
            std::make_shared<std::vector<float>>(std::vector<float>(s_info.img_width*s_info.img_height));
        for (int x = 0; x < s_info.img_width; x++)
        {
            for (int y = 0; y < s_info.img_height; y++)
            {
                int cont_i = y * s_info.img_width + x;
                float_d_buffer->at(cont_i) = d_float->FloatValueAt(x, y).second;
            }
        }
        s_info.depth_buffer = float_d_buffer;

        if (s_info.triangle_buffer == nullptr)
        {
            std::shared_ptr<std::vector<shader::vertex_data>> vdata_triangle_buffer = 
                std::make_shared<std::vector<shader::vertex_data>>(std::vector<shader::vertex_data>(triangles->size() * 3));

            int linear_tr_i = 0;
            for (int i = 0; i < triangles->size() * 3; i+=3)
            {
                vdata_triangle_buffer->at(i) = shader::vertex_data { 
                    (float)vertices->at(triangles->at(linear_tr_i)[0])[0],
                    (float)vertices->at(triangles->at(linear_tr_i)[0])[1],
                    (float)vertices->at(triangles->at(linear_tr_i)[0])[2],
                    0.f
                };

                vdata_triangle_buffer->at(i + 1) = shader::vertex_data { 
                    (float)vertices->at(triangles->at(linear_tr_i)[1])[0],
                    (float)vertices->at(triangles->at(linear_tr_i)[1])[1],
                    (float)vertices->at(triangles->at(linear_tr_i)[1])[2],
                    0.f
                };

                vdata_triangle_buffer->at(i + 2) = shader::vertex_data { 
                    (float)vertices->at(triangles->at(linear_tr_i)[2])[0],
                    (float)vertices->at(triangles->at(linear_tr_i)[2])[1],
                    (float)vertices->at(triangles->at(linear_tr_i)[2])[2],
                    0.f
                };

                // float_triangle_buffer[i]     = (float)vertices->at(triangles->at(linear_tr_i)[0])[0];
                // float_triangle_buffer[i + 1] = (float)vertices->at(triangles->at(linear_tr_i)[0])[1];
                // float_triangle_buffer[i + 2] = (float)vertices->at(triangles->at(linear_tr_i)[0])[2];

                // float_triangle_buffer[i + 3] = (float)vertices->at(triangles->at(linear_tr_i)[1])[0];
                // float_triangle_buffer[i + 4] = (float)vertices->at(triangles->at(linear_tr_i)[1])[1];
                // float_triangle_buffer[i + 5] = (float)vertices->at(triangles->at(linear_tr_i)[1])[2];

                // float_triangle_buffer[i + 6] = (float)vertices->at(triangles->at(linear_tr_i)[2])[0];
                // float_triangle_buffer[i + 7] = (float)vertices->at(triangles->at(linear_tr_i)[2])[1];
                // float_triangle_buffer[i + 8] = (float)vertices->at(triangles->at(linear_tr_i)[2])[2];

                linear_tr_i++;
            }

            s_info.triangle_buffer = vdata_triangle_buffer;
        }
    }
    
    const std::shared_ptr<std::vector<shader::pixel_data>> pdata_vec = shader::dispatch_compute_shader(s_info);

    /**
     *  TODO:
     *      only apply color if normalized dot(dir, normal) large enough
     */

    const Eigen::Vector3d camera_pos (camera_transform(0, 3), camera_transform(1, 3), camera_transform(2, 3));

    std::atomic_int success_count = 0;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < pdata_vec->size(); i++)
    {
        shader::pixel_data pdata = pdata_vec->at(i);

        const Eigen::Vector3d hit_point = Eigen::Vector3d(pdata.p_x, pdata.p_y, pdata.p_z);
        const int tr_i = pdata.tr_i;
        
        if (tr_i >= 0)
        {
            const Eigen::Vector2d point_uv = uv::get_point_uv(
                    vertices->at(triangles->at(tr_i)[0]),
                    vertices->at(triangles->at(tr_i)[1]),
                    vertices->at(triangles->at(tr_i)[2]),
                    hit_point,
                    triangle_uvs->at(tr_i)[0],
                    triangle_uvs->at(tr_i)[1],
                    triangle_uvs->at(tr_i)[2]
                );

            const int uv_x = (int)((float)texture.width_ * point_uv[0]);
            const int uv_y = (int)((float)texture.height_ * point_uv[1]);

            if (uv_x < 0 || uv_x > texture.width_ || uv_y < 0 || uv_y > texture.height_)
            {
                continue;
            }

            const Eigen::Vector3d v0 = vertices->at(triangles->at(tr_i)[0]);
            const Eigen::Vector3d v1 = vertices->at(triangles->at(tr_i)[1]);
            const Eigen::Vector3d v2 = vertices->at(triangles->at(tr_i)[2]);

            // only project if perpendicular (within threshold)

            // normalize(cross(v1 - v0, v2 - v0));
            const Eigen::Vector3d tr_normal = ((v2 - v0).cross(v1 - v0)).normalized();
            const Eigen::Vector3d ray_dir = (hit_point - camera_pos).normalized();

            // std::cout << "dot projection " << tr_normal.dot(ray_dir) << "\n";
            
            // use dot product to determine if the projection is allowed,
            // i.e. reject if the projection ray comes at a shallow angle at the triangle face
            if (tr_normal.dot(ray_dir) < 1.0 - ray_threshold)
                continue;

            int x = pdata.coord_x;
            int y = pdata.coord_y;
            const auto sample_to_r = texture.PointerAt<uint8_t>(uv_x, uv_y, 0);
            const auto sample_to_g = texture.PointerAt<uint8_t>(uv_x, uv_y, 1);
            const auto sample_to_b = texture.PointerAt<uint8_t>(uv_x, uv_y, 2);

            /**
             *  TODO: implement a way of finding the best view to project colors from
             */

            // if (*sample_to_r != 0.0f && *sample_to_b != 0.0f && *sample_to_g != 0.0f)
            //     continue;

            const uint8_t red = *(rgb.PointerAt<uint8_t>(x, y, 0));
            const uint8_t green = *(rgb.PointerAt<uint8_t>(x, y, 1));
            const uint8_t blue = *(rgb.PointerAt<uint8_t>(x, y, 2));

            *sample_to_r = red;
            *sample_to_g = green;
            *sample_to_b = blue;

            success_count++;
        }
    }

    std::cout << "succesfull triangles projected " << success_count << "\n\n";

    // texture = *texture.FlipVertical();
    // o3d::io::WriteImage(DATAFILE_PATH + "texture_out_gpu.png", texture);
}

o3d::geometry::Image non_overwrite_blend_images(const std::vector<o3d::geometry::Image>& textures)
{
    /**
     * TODO:
     *     add  pixel filtering based on frame depth, 
     *     only project if parallel enough
     */
    
    o3d::geometry::Image blend_texture = textures.at(0);

    for (int i = 0; i < textures.size(); i++)
    {
        for (int x = 0; x < blend_texture.width_; x++)
        {
            for (int y = 0; y < blend_texture.height_; y++)
            {
                const uint8_t r = *blend_texture.PointerAt<uint8_t>(x, y, 0);
                const uint8_t g = *blend_texture.PointerAt<uint8_t>(x, y, 1);
                const uint8_t b = *blend_texture.PointerAt<uint8_t>(x, y, 2);

                if (r == 0 && g == 0 && b == 0)
                {
                    *blend_texture.PointerAt<uint8_t>(x, y, 0) = *textures.at(i).PointerAt<uint8_t>(x, y, 0);
                    *blend_texture.PointerAt<uint8_t>(x, y, 1) = *textures.at(i).PointerAt<uint8_t>(x, y, 1);
                    *blend_texture.PointerAt<uint8_t>(x, y, 2) = *textures.at(i).PointerAt<uint8_t>(x, y, 2);
                }
            }
        }
    }

    return blend_texture;
}

void project_texture_on_mesh(o3d::geometry::Image& rgb, o3d::geometry::Image d, 
        const o3d::camera::PinholeCameraIntrinsic& camera_intr, const float depth_scale,
        surface::SurfaceMesh& mesh, o3d::geometry::Image& texture)
{
    // NOTE: this is most likely broken, probably not worth investigating

    const int width = rgb.width_;
    const int height = rgb.height_;

    const Eigen::Vector3d camera_position(0.0, 0.0, 0.0);

    const std::vector<Eigen::Vector3d>* vertices = mesh.get_vertices();
    const std::vector<Eigen::Vector3i>* triangles = mesh.get_triangles();
    const std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* triangle_uvs = mesh.get_triangle_uvs();

    std::atomic_int success_count = 0;
    std::atomic_int not_found = 0;
    std::atomic_int d_val_zero = 0;
    Timer t;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const float d_val = (float)(*d.PointerAt<uint16_t>(x, y)) * depth_scale;
            if (math::is_close(d_val, 0.f))
            {
                d_val_zero++;
                continue;
            }

            const Eigen::Vector3d point = math::pixel_to_world(Eigen::Vector2d((double)x, (double)y), d_val, camera_intr);

            auto [hit_point, success, tr_i] = math::raycast_from_point_to_surface(
                camera_position, point.normalized(), *mesh.get_vertices(), *mesh.get_triangles(), *mesh.get_triangle_normals());
            
            if (tr_i == -1)
            {
                not_found++;
                continue;
            }

            Eigen::Vector2d point_uv = uv::get_point_uv(
                vertices->at(triangles->at(tr_i)[0]),
                vertices->at(triangles->at(tr_i)[1]),
                vertices->at(triangles->at(tr_i)[2]),
                hit_point,
                triangle_uvs->at(tr_i)[0],
                triangle_uvs->at(tr_i)[1],
                triangle_uvs->at(tr_i)[2]
            );

            const int uv_x = (int)((float)texture.width_ * point_uv[0]);
            const int uv_y = (int)((float)texture.height_ * point_uv[1]);

            if (uv_x < 0 || uv_x > texture.width_ || uv_y < 0 || uv_y > texture.height_)
            {
                continue;
            }

            const auto sample_to_r = texture.PointerAt<uint8_t>(uv_x, uv_y, 0);
            const auto sample_to_g = texture.PointerAt<uint8_t>(uv_x, uv_y, 1);
            const auto sample_to_b = texture.PointerAt<uint8_t>(uv_x, uv_y, 2);
            const uint8_t red = *rgb.PointerAt<uint8_t>(x, y, 0);
            const uint8_t green = *rgb.PointerAt<uint8_t>(x, y, 1);
            const uint8_t blue = *rgb.PointerAt<uint8_t>(x, y, 2);

            *sample_to_r = red;
            *sample_to_g = green;
            *sample_to_b = blue;
            
            success_count++;
        }

        std::cout << y << " y val\n";
    }
    t.stop();

    std::cout << "succesfull triangles projected " << success_count << "\n";
    std::cout << not_found << " of not found, " << d_val_zero << " of d_val_zero, " << not_found + success_count + d_val_zero << " of total\n";

    texture = *texture.FlipVertical();
    o3d::io::WriteImage(DATAFILE_PATH + "texture_out.png", texture);

}



}
}
