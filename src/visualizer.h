#pragma once

#include <Open3D/Open3D.h>
#include <memory>
#include <stdexcept>
#include "constants.h"


namespace stitcher3d
{


class Visualizer
{
public:
    static Visualizer* get_instance() { if (m_instance == nullptr) m_instance = new Visualizer(); return m_instance; }

    void update(const std::shared_ptr<o3d::geometry::Geometry3D> to_update);

    void update_gui();

    void clear();

private:
    Visualizer() :
        m_width(1280),
        m_height(720),
        m_pointcloud(std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud())),
        m_trianglemesh(std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh())),
        viz(o3d::visualization::Visualizer()),
        viz_init(false)
     { }

    static Visualizer* m_instance;

    int m_width, m_height;

    std::shared_ptr<o3d::geometry::PointCloud> m_pointcloud;
    std::shared_ptr<o3d::geometry::TriangleMesh> m_trianglemesh;

    o3d::visualization::Visualizer viz;

    bool viz_init;

};

}