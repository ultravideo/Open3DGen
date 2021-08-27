#pragma once

#include <iostream>
#include <memory>
#include <exception>

#include "constants.h"
// #include "pose_graph.h"
#include "poseg2.h"
#include "s3d_camera.h"


namespace stitcher3d
{


class RealMesh
{

public:
	RealMesh(PoseGraph* pgraph);
	~RealMesh();

	void add_frame(Camera& cam);


private:
	PoseGraph* pgraph;
	double pointcloud_scale;

	std::vector<std::shared_ptr<o3d::geometry::PointCloud>> pointclouds;
	std::shared_ptr<o3d::geometry::PointCloud> totalcloud;

};
	
}