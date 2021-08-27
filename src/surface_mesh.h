#pragma once

#include <Open3D/Geometry/BoundingVolume.h>
#include <Open3D/Open3D.h>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <array>
#include <vector>
#include <numeric>
#include <exception>
#include "visualizer.h"
#include "constants.h"
// #include "pose_graph.h"
#include "poseg2.h"
// #include "../Open3D/3rdparty/tinyobjloader/tiny_obj_loader.h"
#include "../libraries/tinyobj/tiny_obj_loader.h"


namespace stitcher3d
{
namespace surface
{

class SurfaceMesh
{
public:
    SurfaceMesh();
    ~SurfaceMesh();

    void add_frame(const std::shared_ptr<o3d::geometry::PointCloud> newcloud);

    void clear();

    std::shared_ptr<o3d::geometry::TriangleMesh> get_mesh();
    std::shared_ptr<o3d::geometry::PointCloud> get_pointcloud();

    void generate_mesh(const uint32_t poisson_depth);

    void write_mesh(const std::string& filepath) const;
    void read_mesh(const std::string& filepath);

    void crop_mesh(o3d::geometry::OrientedBoundingBox bbox);

    void write_pointcloud(const std::string& filepath) const;
    void read_pointcloud(const std::string& filepath);

    bool is_not_empty() const { return m_pointcloud->HasPoints() || m_mesh->HasTriangles(); }

    void post_process_pointcloud();

    void pointclouds_from_posegraph(const PoseGraph& pg);

    void pointclouds_from_clouds(const std::vector<std::shared_ptr<o3d::geometry::PointCloud>>& clouds);

    void generate_grouped_triangle_uvs();

    std::vector<Eigen::Vector3i>* get_triangles() { return &m_triangles; }
    std::vector<Eigen::Vector3d>* get_vertices() { return &m_vertices; }
    std::vector<Eigen::Vector3d>* get_triangle_normals() 
        { if (m_triangle_normals.size() == 0) m_mesh->ComputeTriangleNormals(); m_triangle_normals = m_mesh->triangle_normals_;
            return &m_triangle_normals; }

    std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* get_triangle_uvs() 
        { return &m_triangle_uvs; }

    std::vector<Eigen::Vector2d>* get_flat_triangle_uvs() 
        { return &m_triangle_flat_uvs; }

    void flatten_uvs();

    void decimate(const float voxel_size);

    std::vector<double> get_flatten_vertices() const;
    std::vector<uint32_t> get_flatten_indices() const;

    void scale(const double scaling_factor)
    {
        if (m_pointcloud != nullptr)
            m_pointcloud->Scale(scaling_factor);

        if (m_mesh != nullptr)
            m_mesh->Scale(scaling_factor);
    }

    static inline std::shared_ptr<surface::SurfaceMesh> mesh_from_cameras_and_obj(
        const std::vector<std::shared_ptr<Camera>>& cameras, const double pointcloud_scale,
        const std::string& obj_path)
    {
        std::vector<std::shared_ptr<o3d::geometry::PointCloud>> clouds;
        std::vector<std::shared_ptr<const o3d::geometry::Geometry>> debug_clouds;

        for (const auto cam : cameras)
        {
            auto cloud = rgbd::create_pcloud_from_rgbd(*cam->rgb, *cam->depth, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, false, cam->intr);
            cloud->Scale(pointcloud_scale, false);
            cloud->Transform(cam->T);

            clouds.push_back(cloud);
            debug_clouds.push_back(cloud);
        }

        // o3d::visualization::DrawGeometries(debug_clouds);

        std::shared_ptr<surface::SurfaceMesh> mesh = std::make_shared<surface::SurfaceMesh>(surface::SurfaceMesh());
        mesh->pointclouds_from_clouds(clouds);

        std::shared_ptr<o3d::geometry::TriangleMesh> obj_mesh;

        // a path was given, load the mesh
        if (obj_path != "")
        {
            obj_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            const bool status = o3d::io::ReadTriangleMeshFromOBJ(obj_path, *obj_mesh, true);
            mesh->m_mesh = obj_mesh;
            mesh->m_triangles = mesh->m_mesh->triangles_;
            mesh->m_vertices = mesh->m_mesh->vertices_;
            mesh->m_triangle_normals = mesh->m_mesh->triangle_normals_;
            mesh->m_triangle_flat_uvs = mesh->m_mesh->triangle_uvs_;

            if (mesh->m_triangle_flat_uvs.size() != 0)
            {
                mesh->generate_grouped_triangle_uvs();

                mesh->flatten_uvs();
                mesh->m_mesh->triangle_uvs_ = *mesh->get_flat_triangle_uvs();
            }

            mesh->mesh_file_path = obj_path;
        }

        return mesh;
    }

    void generate_empty_uvs();

    void set_flat_uvs(const std::vector<double>& flat_uvs);

    /**
     *  a hacky workaround in order to properly use xatlas for
     *  uv unwrapping
     */
    tinyobj2::mesh_t get_as_tinyobj_mesh() const;

protected:

    std::string mesh_file_path = "mesh_temp.obj";

    std::shared_ptr<o3d::geometry::TriangleMesh> m_mesh;
    std::shared_ptr<o3d::geometry::PointCloud> m_pointcloud;

    std::vector<Eigen::Vector3i> m_triangles;
    std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector3d> m_triangle_normals;
    std::vector<Eigen::Vector2d> m_triangle_flat_uvs;
    std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>> m_triangle_uvs;
    // std::vector<Eigen::Vector3d> m_vertex_colors;

    double m_avg_point_distance;

    void calculate_avg_point_distance();


};


}
}