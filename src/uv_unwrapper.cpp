#include "uv_unwrapper.h"
#include <Eigen/src/Core/Matrix.h>
#include <boost/smart_ptr/detail/local_counted_base.hpp>
#include <cstdint>
#include <functional>
#include <sys/types.h>
#include <vector>

namespace stitcher3d
{
namespace uv
{

void unwrap_uvs_xatlas(std::shared_ptr<surface::SurfaceMesh> mesh)
{
    xatlas::Atlas* uvatlas = xatlas::Create();
    xatlas::MeshDecl mesh_decl;

    mesh->generate_empty_uvs();
    const auto tomesh = mesh->get_as_tinyobj_mesh();

    // vertices
    mesh_decl.vertexCount = (uint32_t)tomesh.positions.size() / 3;
    mesh_decl.vertexPositionData = tomesh.positions.data();
    mesh_decl.vertexPositionStride = sizeof(float) * 3;

    if (!tomesh.normals.empty())
    {
        mesh_decl.vertexNormalData = tomesh.normals.data();
        mesh_decl.vertexNormalStride = sizeof(float) * 3;
    }

    mesh_decl.indexCount = (uint32_t)tomesh.indices.size();
    mesh_decl.indexData = tomesh.indices.data();
    mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

    std::cout << "\n";
    std::cout << "num of vertices: " << tomesh.positions.size() << "\n";
    std::cout << "num of indices: " << tomesh.indices.size() << "\n";
    std::cout << "\n";

    if (tomesh.num_vertices.size() != tomesh.indices.size() / 3)
    {
        mesh_decl.faceVertexCount = tomesh.num_vertices.data();
        // mesh_decl.faceCount = (uint32_t)tomesh.num_vertices.size();
        mesh_decl.faceCount = (uint32_t)(tomesh.indices.size() / 3);
    }
    
    xatlas::AddMeshError err = xatlas::AddMesh(uvatlas, mesh_decl, 1);
    if (err != xatlas::AddMeshError::Success)
    {
        std::cout << "xatlas encountered an error: " << xatlas::StringForEnum(err) << "\n";
        return;
    }

    xatlas::PackOptions po;
    po.bruteForce = true;
    // po.blockAlign = true;

    xatlas::ChartOptions co;
    co.fixWinding = true;
    co.maxIterations = 10;
    co.maxCost = 200.;

    xatlas::Generate(uvatlas, co, po);
    std::cout << "xatlas uv unwrapping succesful\n";

    std::vector<Eigen::Vector2d> uv_vec;
    uv_vec.reserve(tomesh.positions.size());

    // std::cout << "meshcount: " << uvatlas->meshCount << "\n";

    const xatlas::Mesh& xm = uvatlas->meshes[0];
    for (uint32_t i = 0; i < xm.vertexCount; i++)
    {
        const xatlas::Vertex& vert = xm.vertexArray[i];
        // uv_coords.push_back(vert.uv[0] / (double)uvatlas->width);
        // uv_coords.push_back(vert.uv[1] / (double)uvatlas->height);

        uv_vec.push_back(Eigen::Vector2d(
                vert.uv[0] / (double)uvatlas->width,
                vert.uv[1] / (double)uvatlas->height
            ));
    }

    uv_vec.shrink_to_fit();
    std::cout << "xm vcount: " << xm.vertexCount << "\n";
    std::cout << "uv coords size: " << uv_vec.size() << "\n";

    pad_uvs(uv_vec, tomesh.indices, UV_PAD_SCALE);
    const std::vector<double> uv_coords = flatten_vec_uvs(uv_vec);

    mesh->set_flat_uvs(uv_coords);
    xatlas::Destroy(uvatlas);
}

void pad_uvs(std::vector<Eigen::Vector2d>& uv_coords, const std::vector<uint32_t>& indices, const double pad_scale)
{
    /**
     *  scale the uv triangles to be pad_scale times smaller,
     *      towards the top-left corner of the uv triangle
     */

    for (int ii = 0; ii < indices.size(); ii += 3)
    {
        Eigen::Vector2d& uv0 = uv_coords[indices[ii]];
        Eigen::Vector2d& uv1 = uv_coords[indices[ii + 1]];
        Eigen::Vector2d& uv2 = uv_coords[indices[ii + 2]];

        const Eigen::Vector2d refpos = Eigen::Vector2d(
            min(uv0.x(), min(uv1.x(), uv2.x())),
            min(uv0.y(), min(uv1.y(), uv2.y()))
            );

        uv0 -= refpos;
        uv1 -= refpos;
        uv2 -= refpos;

        uv0 *= pad_scale;
        uv1 *= pad_scale;
        uv2 *= pad_scale;

        uv0 += refpos;
        uv1 += refpos;
        uv2 += refpos;
    }
}


void calculate_individual_uvs(std::shared_ptr<surface::SurfaceMesh> mesh, const float uv_margin)
{
    std::vector<Eigen::Vector3i>* triangles = mesh->get_triangles();
    std::vector<Eigen::Vector3d>* vertices = mesh->get_vertices();
    std::vector<Eigen::Vector3d>* triangle_normals = mesh->get_triangle_normals();
    std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* uvs = mesh->get_triangle_uvs();

    uvs->reserve(triangles->size());

    for (int i = 0; i < triangles->size(); i++)
    {
        const Eigen::Vector3d v0 = vertices->at(triangles->at(i).coeff(0));
        const Eigen::Vector3d v1 = vertices->at(triangles->at(i).coeff(1));
        const Eigen::Vector3d v2 = vertices->at(triangles->at(i).coeff(2));

        const Eigen::Vector3d tr_n = triangle_normals->at(i);

        const Eigen::Quaterniond q_rot = math::quaternion_look_rotation(tr_n).inverse();

        const Eigen::Vector3d uv0 = q_rot * (v0 * UV_PRE_SCALAR);
        const Eigen::Vector3d uv1 = q_rot * (v1 * UV_PRE_SCALAR);
        const Eigen::Vector3d uv2 = q_rot * (v2 * UV_PRE_SCALAR);

        const std::array<Eigen::Vector2d, TR_VERT_COUNT> uv
        {
            Eigen::Vector2d(uv0.x(), uv0.y()),
            Eigen::Vector2d(uv1.x(), uv1.y()),
            Eigen::Vector2d(uv2.x(), uv2.y())
        };

        uvs->push_back(uv);
    }

    const std::vector<unsigned int> area_sorted_idx = sort_uv_triangles_by_area(uvs);

    pack_uvs(uvs, uv_margin, area_sorted_idx);
    mesh->flatten_uvs();

    mesh->get_mesh()->triangle_uvs_ = *mesh->get_flat_triangle_uvs();
}

Eigen::Vector2d get_point_uv(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& p2, const Eigen::Vector3d& p,
                            const Eigen::Vector2d& uv0, const Eigen::Vector2d& uv1,
                            const Eigen::Vector2d& uv2)
{
    const auto f1 = p0 - p;
    const auto f2 = p1 - p;
    const auto f3 = p2 - p;

    const double a = (p0 - p1).cross(p0 - p2).norm();
    const double a1 = f2.cross(f3).norm() / a;
    const double a2 = f3.cross(f1).norm() / a;
    const double a3 = f1.cross(f2).norm() / a;

    return uv0 * a1 + uv1 * a2 + uv2 * a3;
}

std::vector<unsigned int> sort_uv_triangles_by_area(
            const std::vector<std::array<Eigen::Vector2d, TR_VERT_COUNT>>* uvs)
{
    std::vector<float> triangle_areas;
    triangle_areas.reserve(uvs->size());
    for (const std::array<Eigen::Vector2d, TR_VERT_COUNT>& uv : *uvs)
    {
        triangle_areas.emplace_back(
            std::abs(
                uv[0][0] * (uv[1][1] - uv[2][1]) + uv[1][0] * (uv[2][1] - uv[0][1]) + uv[2][0] * (uv[0][1] - uv[1][1])
            ) / 2.f
        );
    }

    std::vector<unsigned int> idx(triangle_areas.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
        [&triangle_areas] (size_t i1, size_t i2)
        {
            return triangle_areas[i1] > triangle_areas[i2];
        }
    );

    return idx;
}

}
}