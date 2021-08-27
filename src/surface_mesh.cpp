#include "surface_mesh.h"
#include <Open3D/Geometry/BoundingVolume.h>
#include <Open3D/Geometry/TriangleMesh.h>
#include <memory>

namespace stitcher3d
{
namespace surface
{

SurfaceMesh::SurfaceMesh() :
    m_mesh(std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh())),
    m_pointcloud(std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud())),
    m_triangles(std::vector<Eigen::Vector3i>()),
    m_vertices(std::vector<Eigen::Vector3d>()),
    m_triangle_normals(std::vector<Eigen::Vector3d>()),
    m_triangle_flat_uvs(std::vector<Eigen::Vector2d>()),
    m_avg_point_distance(0)
{

}

SurfaceMesh::~SurfaceMesh() { }

void SurfaceMesh::pointclouds_from_posegraph(const PoseGraph& pg)
{
    for (auto cloud : pg.depth_pointclouds)
    {
        add_frame(cloud);
    }
}

void SurfaceMesh::pointclouds_from_clouds(const std::vector<std::shared_ptr<o3d::geometry::PointCloud>>& clouds)
{
    for (const auto cloud : clouds)
    {
        add_frame(cloud);
    }
}

void SurfaceMesh::crop_mesh(o3d::geometry::OrientedBoundingBox bbox)
{
    // a hacky workaround, because Crop() doesn't work unless the bbox rotation is identity
    const Eigen::Matrix3d R = bbox.R_;
    m_mesh->Rotate(R.inverse());
    bbox.Rotate(R.inverse());

    auto new_mesh = m_mesh->Crop(bbox);
    new_mesh->Rotate(R);

    new_mesh->RemoveDuplicatedVertices();
    new_mesh->RemoveUnreferencedVertices();
    new_mesh->RemoveDegenerateTriangles();
    
    m_mesh = new_mesh;
    m_triangles = m_mesh->triangles_;
    m_vertices = m_mesh->vertices_;
}


void SurfaceMesh::add_frame(const std::shared_ptr<o3d::geometry::PointCloud> newcloud)
{
    if (!m_pointcloud->HasPoints())
    {
        *m_pointcloud += *newcloud;
        // calculate_avg_point_distance();
        return;
    }
    
    o3d::geometry::PointCloud tempcloud(*newcloud);

    // remove overlapping points
    // std::vector<double> dist_vec = tempcloud.ComputePointCloudDistance(*m_pointcloud);
    // std::vector<size_t> index_vec;
    // size_t v_index = 0;
    // double stitch_threshold = m_avg_point_distance * STITCH_DISTANCE_MULTIPLIER;
    // std::for_each(dist_vec.begin(), dist_vec.end(), [&index_vec, &v_index, stitch_threshold](const double& value)
    // {
    //     if (value < stitch_threshold)
    //         index_vec.push_back(v_index);

    //     v_index++;
    // });

    // Open3D 0.10.x API call
    // tempcloud = *tempcloud.SelectByIndex(index_vec, true);
    // 0.9.0

    // tempcloud = *tempcloud.SelectDownSample(index_vec, true);

    *m_pointcloud += tempcloud;
    // calculate_avg_point_distance();
}

void SurfaceMesh::decimate(const float voxel_size)
{
    if (m_mesh == nullptr)
        throw std::runtime_error("m_mesh was nullptr, aborting in decimate()");

    std::cout << "triangle count before decimation: " << m_mesh->triangles_.size() << "\n";

    // m_mesh = m_mesh->SimplifyQuadricDecimation(target_vertex_amount);
    m_mesh = m_mesh->SimplifyVertexClustering(voxel_size);
    
    m_mesh->RemoveDuplicatedVertices();
    m_mesh->RemoveUnreferencedVertices();
    m_mesh->RemoveDegenerateTriangles();

    std::cout << "triangle count after decimation: " << m_mesh->triangles_.size() << "\n";

    m_triangles = m_mesh->triangles_;
    m_vertices = m_mesh->vertices_;
}

std::shared_ptr<o3d::geometry::PointCloud> SurfaceMesh::get_pointcloud()
{
    return m_pointcloud;
}

std::shared_ptr<o3d::geometry::TriangleMesh> SurfaceMesh::get_mesh()
{
    return m_mesh;
}

void SurfaceMesh::write_pointcloud(const std::string& filepath) const
{
    o3d::io::WritePointCloud(filepath, *m_pointcloud);
}

void SurfaceMesh::read_pointcloud(const std::string& filepath)
{
    if (!o3d::io::ReadPointCloud(filepath, *m_pointcloud))
        throw std::runtime_error("couldn't read file " + filepath + ", aborting in read_pointcloud()");
}

void SurfaceMesh::generate_empty_uvs()
{
    const size_t uvcount = m_mesh->triangles_.size() * 3;
    m_triangle_normals.clear();
    m_triangle_flat_uvs.reserve(uvcount);

    for (int i = 0; i < uvcount; i++)
        m_triangle_flat_uvs.push_back(Eigen::Vector2d(0, 0));

    m_mesh->triangle_uvs_ = m_triangle_flat_uvs;
}

void SurfaceMesh::write_mesh(const std::string& filepath) const
{
    std::cout << "writing mesh to: " << filepath << "\n";
    std::cout << "tr uvs: " << m_mesh->triangle_uvs_.size() << "\n";
    std::cout << "vertices: " << m_mesh->vertices_.size() << "\n";
    std::cout << "triangles: " << m_mesh->triangles_.size() << "\n";

    std::cout << "begin writing mesh to file\n"; 
    if (!o3d::io::WriteTriangleMesh(filepath, *m_mesh, true, false, true, false, true, false))
        throw std::runtime_error("couldn't write mesh " + filepath + " to disk, aborting in write_mesh()");
    std::cout << "mesh written to file: " << filepath << "\n";
}

void SurfaceMesh::read_mesh(const std::string& filepath)
{
    if (!o3d::io::ReadTriangleMesh(filepath, *m_mesh))
        throw std::runtime_error("couldn't read mesh " + filepath + " from disk, aborting in read_mesh()");

    m_triangles = m_mesh->triangles_;
    m_vertices = m_mesh->vertices_;
    // m_triangle_normals = m_mesh->triangle_normals_;
    m_triangle_flat_uvs = m_mesh->triangle_uvs_;
}

void SurfaceMesh::generate_grouped_triangle_uvs()
{
    if (m_triangle_flat_uvs.size() == 0)
        throw std::runtime_error("no m_triangle_flat_uvs, aborting in generate_grouped_triangle_uvs");

    // m_triangle_uvs.clear();
    // for (int i = 0; i < m_triangles.size(); i++)
    // {
    //     const Eigen::Vector3i tr = m_triangles[i];
    //     m_triangle_uvs.emplace_back(std::array<Eigen::Vector2d, TR_VERT_COUNT>{ m_triangle_flat_uvs[tr.x()], m_triangle_flat_uvs[tr.y()], m_triangle_flat_uvs[tr.z()] });
    // }

    m_triangle_uvs.clear();
    for (int i = 0; i < m_triangle_flat_uvs.size(); i += 3)
    {
        m_triangle_uvs.emplace_back(std::array<Eigen::Vector2d, TR_VERT_COUNT>{ m_triangle_flat_uvs[i], m_triangle_flat_uvs[i+1], m_triangle_flat_uvs[i+2] });
    }
}

void SurfaceMesh::generate_mesh(const uint32_t poisson_depth)
{
    if (m_pointcloud == nullptr)
        throw std::runtime_error("m_pointcloud was nullptr, aborting in generate_mesh()");

    // downscale the pointcloud, otherwise it's going to take forever. 5mm voxel size better 
    // be good enough
    // const auto downsample_cloud = m_pointcloud->VoxelDownSample(0.005);
    // m_pointcloud = downsample_cloud;

    auto [new_mesh, trash] = o3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*m_pointcloud, poisson_depth);
    new_mesh = new_mesh->FilterSmoothLaplacian(LAPLACIAN_ITERATIONS, LAPLACIAN_LAMBDA);

    // calculate_avg_point_distance();
    // auto new_mesh = o3d::geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*m_pointcloud, {m_avg_point_distance * 3.0, m_avg_point_distance});
    
    m_mesh = new_mesh;

    m_triangles = m_mesh->triangles_;
    m_vertices = m_mesh->vertices_;
}

void SurfaceMesh::calculate_avg_point_distance()
{
    std::vector<double> distances = m_pointcloud->ComputeNearestNeighborDistance();
    m_avg_point_distance = std::accumulate(distances.begin(), distances.end(), (double)0) / (double)distances.size();
}

void SurfaceMesh::post_process_pointcloud()
{
    calculate_avg_point_distance();

    std::shared_ptr<o3d::geometry::PointCloud> tempcloud_ptr;
    std::vector<size_t> a;

    if (m_avg_point_distance > 0.0)
    {
        std::tie(tempcloud_ptr, a) = m_pointcloud->RemoveRadiusOutliers(PP_OUTLIER_NB_POINTS, 
                m_avg_point_distance * PP_OUTLIER_MULTIPLIER);
    }
    else
    {
        tempcloud_ptr = std::make_shared<o3d::geometry::PointCloud>(*m_pointcloud);
    }
    
    tempcloud_ptr = tempcloud_ptr->VoxelDownSample(PP_VOXEL_SIZE);
    m_pointcloud = tempcloud_ptr;
}

void SurfaceMesh::flatten_uvs()
{
    if (m_triangle_uvs.size() == 0)
        throw std::runtime_error("SurfaceMesh m_triangle_uvs length was 0, aborting in flatten_uvs()!");

    m_triangle_flat_uvs.clear();
    m_triangle_flat_uvs.reserve(m_triangle_uvs.size() * 3);

    for (const std::array<Eigen::Vector2d, TR_VERT_COUNT>& tr_uv : m_triangle_uvs)
    {
        m_triangle_flat_uvs.emplace_back(tr_uv[0]);
        m_triangle_flat_uvs.emplace_back(tr_uv[1]);
        m_triangle_flat_uvs.emplace_back(tr_uv[2]);
    }
}

std::vector<double> SurfaceMesh::get_flatten_vertices() const
{
    std::vector<double> flat_verts;
    flat_verts.reserve(m_vertices.size() * 3);

    for (int i = 0; i < m_vertices.size(); i++)
    {
        const auto v = m_vertices[i];
        flat_verts.push_back(v.x());
        flat_verts.push_back(v.y());
        flat_verts.push_back(v.z());
    }

    return flat_verts;
}

void SurfaceMesh::set_flat_uvs(const std::vector<double>& flat_uvs)
{
    m_triangle_flat_uvs.clear();
    m_triangle_flat_uvs.reserve(flat_uvs.size() / 2);

    for (int i = 0; i < flat_uvs.size(); i += 2)
    {
        const Eigen::Vector2d uv (flat_uvs[i], flat_uvs[i + 1]);
        m_triangle_flat_uvs.push_back(uv);
    }

    generate_grouped_triangle_uvs();
    m_mesh->triangle_uvs_ = m_triangle_flat_uvs;
}

std::vector<uint32_t> SurfaceMesh::get_flatten_indices() const
{
    std::vector<uint32_t> flat_i;
    flat_i.reserve(m_triangles.size() * 3);

    for (int i = 0; i < m_triangles.size(); i++)
    {
        const auto v = m_triangles[i];
        flat_i.push_back(v.x());
        flat_i.push_back(v.y());
        flat_i.push_back(v.z());
    }

    return flat_i;
}

tinyobj2::mesh_t SurfaceMesh::get_as_tinyobj_mesh() const
{
    const std::string temp_path = utilities::split_string(mesh_file_path, '.')[0] + "_temp.obj";
    write_mesh(temp_path);

    std::vector<tinyobj2::shape_t> shapes;
	std::vector<tinyobj2::material_t> materials;
	std::string err;

    if (!tinyobj2::LoadObj(shapes, materials, err, temp_path.c_str(), nullptr, 0))
        throw std::runtime_error("failed to load obj, " + err);

    if (shapes.size() != 1)
        throw std::runtime_error("invalid shape size for this implementation: " + std::to_string(shapes.size()));

    return shapes.front().mesh;
}

}
}