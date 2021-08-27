#include "visualizer.h"

namespace stitcher3d
{

Visualizer* Visualizer::m_instance = 0;

void Visualizer::update(const std::shared_ptr<o3d::geometry::Geometry3D> to_update)
{
    if (!viz_init)
    {
        viz.CreateVisualizerWindow("Stither3D Visualizer", m_width, m_height);
        viz_init = true;
    }

    if (to_update->GetGeometryType() == o3d::geometry::Geometry3D::GeometryType::PointCloud)
    {
        auto tmp = std::dynamic_pointer_cast<o3d::geometry::PointCloud>(to_update);;
        if (!viz.HasGeometry())
        {
            m_pointcloud = tmp;
            viz.AddGeometry(m_pointcloud);
        }
        else
        {
            if (tmp != m_pointcloud)
            {
                viz.RemoveGeometry(m_pointcloud);
                m_pointcloud = tmp;
                viz.AddGeometry(m_pointcloud);
            }
            else { viz.UpdateGeometry(m_pointcloud); }
        }
    }
    else if (to_update->GetGeometryType() == o3d::geometry::Geometry3D::GeometryType::TriangleMesh)
    {
        auto tmp = std::dynamic_pointer_cast<o3d::geometry::TriangleMesh>(to_update);;
        if (!viz.HasGeometry())
        {
            m_trianglemesh = tmp;
            viz.AddGeometry(m_trianglemesh);
        }
        else
        {
            if (tmp != m_trianglemesh)
            {
                viz.RemoveGeometry(m_trianglemesh);
                m_trianglemesh = tmp;
                viz.AddGeometry(m_trianglemesh);
            }
            else { viz.UpdateGeometry(m_trianglemesh); }
        }
    }
    else { throw std::runtime_error("not support object type!"); }

    update_gui();

}

void Visualizer::update_gui()
{
    viz.PollEvents();
    viz.UpdateRender();
}

void Visualizer::clear()
{
    if (m_pointcloud != nullptr)
        viz.RemoveGeometry(m_pointcloud);
    else if (m_trianglemesh != nullptr)
        viz.RemoveGeometry(m_trianglemesh);

    m_pointcloud = nullptr;
    m_trianglemesh = nullptr;

    viz.DestroyVisualizerWindow();
    viz_init = false;
}

}