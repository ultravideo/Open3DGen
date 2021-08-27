#include <Open3D/Geometry/BoundingVolume.h>
#include <Open3D/Visualization/Utility/DrawGeometry.h>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Open3D/Open3D.h>
#include <exception>
#include <fstream>
#include <memory>

#include "src/registration_params.h"
#include "src/surface_mesh.h"
#include "src/utilities.h"
#include "src/pointcloud_registration.h"
#include "src/uv_unwrapper.h"
#include "src/timer.h"
#include "src/visualizer.h"
#include "src/texture_projector.h"
#include "src/shader_stuff.h"
#include "src/constants.h"

using namespace stitcher3d;

int main(int argc, char* argv[])
{
    const auto [
        input_path, intr_path, output_path, out_res, output_nth, poisson_depth,
        dist_coeffs, cchain_len, lam_min_count, serialized_cameras_path, assets_path,
        refine_cameras, export_mesh, simplify_voxel_size, do_unwrap, load_mesh_path, 
        skip_stride, workgroup_size, write_interm_images, depth_far_clip, 
        max_feature_count, do_crop, ray_threshold, reject_blur, pipeline_state] = utilities::parse_cmdline_args(argc, argv);

    Timer tt;
    std::cout << "\nbegin\n";

    float pointcloud_world_scale;
    auto intr = stitcher3d::utilities::read_camera_intrinsics(intr_path);
    
    // localize cameras and reconstruct the pointcloud

    std::shared_ptr<surface::SurfaceMesh> mesh = nullptr;
    std::vector<std::shared_ptr<Camera>> cameras;
    float pcloud_scale = 0.f;

    if (pipeline_state == registration::PipelineState::Full || 
        pipeline_state == registration::PipelineState::PointcloudOnly)
    {
        std::tie(mesh, cameras, pcloud_scale) = 
            registration::register_pointcloud_sequence_features(input_path, intr, dist_coeffs, { cchain_len, lam_min_count, refine_cameras, skip_stride, depth_far_clip, max_feature_count, assets_path });

        pointcloud_world_scale = pcloud_scale;

        tt.stop("pointcloud registration");

        // serialize the relevant info of cameras
        {
            // write to file
            std::ofstream outfile;
            outfile.open(output_path + "/serialize_cameras.data");
            outfile.clear();
            outfile << utilities::serialize_cameras_to_str(cameras, pcloud_scale);
            outfile.close();
        }

        o3d::io::WritePointCloudToPLY(output_path + "/reconstructed_point_cloud.ply", *mesh->get_pointcloud());

        mesh->generate_mesh(poisson_depth);
        tt.stop("mesh generation");

        // cut the poisson artefacts out
        if (do_crop)
        {
            // balloon the bounding box outwards to avoid cropping wanted detail
            const auto bbox = std::make_shared<o3d::geometry::OrientedBoundingBox>(mesh->get_pointcloud()->GetOrientedBoundingBox().Scale(BOUNDING_BOX_SCALE_FACTOR, true));
            mesh->crop_mesh(*bbox);

            #ifdef DEBUG_VISUALIZE
            o3d::visualization::DrawGeometries({ mesh->get_mesh(), bbox });
            #endif
        }

        if (simplify_voxel_size != 0.f)
        {
            mesh->decimate(simplify_voxel_size * pcloud_scale);

            #ifdef DEBUG_VISUALIZE
            o3d::visualization::DrawGeometries({ mesh->get_mesh()});
            #endif
        }

        if (do_unwrap)
        {
            uv::unwrap_uvs_xatlas(mesh);
            tt.stop("uv calculations");
        }

        if (export_mesh)
            mesh->write_mesh(output_path + "/reconstructed_mesh.obj");

        if (pipeline_state == registration::PipelineState::PointcloudOnly)
            return 0;
    }

    if (pipeline_state == registration::PipelineState::PointcloudOnly)
        return 0;

    // reconstruct the mesh 
    // std::shared_ptr<surface::SurfaceMesh> mesh = nullptr;
    // std::vector<std::shared_ptr<Camera>> cameras;

    // load the cameras and mesh from file for projection
    if (pipeline_state != registration::PipelineState::Full)
    {
        // read the cameras and mesh (deserialize)
        {
            std::ifstream infile;
            if (serialized_cameras_path.empty())
                infile.open(output_path + "/serialize_cameras.data");
            else
                infile.open(serialized_cameras_path + "/serialize_cameras.data");

            std::string line;
            while (std::getline(infile, line))
            {
                // de-serialize the pcloud scale
                if (line.size() < 15)
                {
                    pointcloud_world_scale = std::stof(line);
                    std::cout << "scale: " << pointcloud_world_scale << "\n";
                    continue;
                }
                cameras.emplace_back(std::make_shared<Camera>(utilities::deserialize_cam_from_str(line)));
                cameras.back()->intr = intr;
            }

            // project_only -> mesh exists, load it
            if (pipeline_state == registration::PipelineState::ProjectOnly)
            {
                mesh = surface::SurfaceMesh::mesh_from_cameras_and_obj(cameras, pointcloud_world_scale,
                    load_mesh_path.empty() ? output_path + "/reconstructed_mesh.obj" : load_mesh_path);
            }
            // mesh_only, mesh doesn't exist, create it from pointclouds
            else
            {
                mesh = surface::SurfaceMesh::mesh_from_cameras_and_obj(cameras, pointcloud_world_scale, "");
                mesh->generate_mesh(poisson_depth);    
            }

            if (simplify_voxel_size != 0.f)
            {
                mesh->decimate(simplify_voxel_size * pcloud_scale);

                #ifdef DEBUG_VISUALIZE
                o3d::visualization::DrawGeometries({ mesh->get_mesh()});
                #endif
            }

            if (do_unwrap)
                uv::unwrap_uvs_xatlas(mesh);

            if (export_mesh)
                mesh->write_mesh(output_path + "/reconstructed_mesh.obj");
        }
    }

    if (pipeline_state == registration::PipelineState::MeshOnly)
        return 0;

    // project the textures

    // o3d::visualization::DrawGeometries({mesh->get_mesh(), o3d::geometry::TriangleMesh::CreateCoordinateFrame()});
    // std::exit(0);

    if (pipeline_state == registration::PipelineState::ProjectOnly || 
        pipeline_state == registration::PipelineState::Full)
    {
        std::cout << "creating temporary projection images\n";
        std::vector<o3d::geometry::Image> textures;
        for (int i = 0; i < cameras.size(); i++)
        {
            textures.emplace_back(o3d::geometry::Image());
            if (out_res == 8192)
                o3d::io::ReadImage(assets_path + RAY_BLANK_IMG_8K, textures.back());
            else if (out_res == 4096)
                o3d::io::ReadImage(assets_path + RAY_BLANK_IMG_4K, textures.back());
            else if (out_res == 2048)
                o3d::io::ReadImage(assets_path + RAY_BLANK_IMG_2K, textures.back());
        }

        Timer t;
        textures::project_texture_sequence_on_mesh_gpu(DEFAULT_DEPTH_SCALE, pointcloud_world_scale, *mesh, textures, cameras, output_nth, workgroup_size, ray_threshold, reject_blur);
        // textures::non_zero_convolve_uv_filter(texture);
        std::cout << "texture projection on GPU took: "; t.stop();

        if (write_interm_images)
        {
            std::cout << "begin writing per-frame projected textures to disk\n";
            for (int i = 0; i < textures.size(); i++)
                o3d::io::WriteImage(output_path + "/texture_out_gpu_" + std::to_string(i) + ".png", textures[i]);
            std::cout << "textures saved to " << output_path + "/texture_out_gpu_#.png\n";
        }

        std::cout << "begin average blending images\n";
        o3d::geometry::Image avg_texture = textures::average_blend_images(textures);
        std::cout << "images average blended\n";

        // non-zero blending doesn't result in particularly good
        // textures with the current texture projection algorithm used
        // std::cout << "begin non-overwrite blending images\n";
        // const o3d::geometry::Image non_overwrite_texture = textures::non_overwrite_blend_images(textures);
        // std::cout << "images blended\n";

        o3d::io::WriteImage(output_path + "/texture_out_gpu_average.png", avg_texture);

        std::cout << "begin dilating texture\n";
        textures::dilate_image(avg_texture);
        std::cout << "texture dilated\n";

        o3d::io::WriteImage(output_path + "/texture_out_gpu_average_dilated.png", avg_texture);
        // o3d::io::WriteImage(output_path + "/texture_out_gpu_non_zero.png", non_overwrite_texture);

        std::cout << "blended images saved to " << output_path + 
            "/texture_out_gpu_non_zero.png and " << output_path + "/texture_out_gpu_average.png\n";
    }

    return 0;
}
