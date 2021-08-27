#include "realtime_mesh.h"

using namespace stitcher3d;


RealMesh::RealMesh(PoseGraph* pgraph_) :
	pgraph(pgraph_),
	pointcloud_scale(0.0),
	pointclouds(std::vector<std::shared_ptr<o3d::geometry::PointCloud>>()),
	totalcloud(nullptr)
{
	if (pgraph == nullptr)
		throw std::runtime_error("posegraph was null!");
}

RealMesh::~RealMesh() {  }

void RealMesh::add_frame(Camera& cam)
{
	/**
	 * 	PIPELINE:
	 * 		- x  calculate pointcloud
	 * 		- x  create a pointcloud
	 * 		- remove overlap
	 * 		- compute normals
	 * 		- compute poisson (CGAL vs o3d?)
	 * 		- remove extra geometry (pointcloud negative mask?)
	 * 		- project textures from view (custom renderer?)
	 * 			-- use headless shader renderer?
	 */
	
	/*if (pgraph->cameras.size() < 2)
	{
		std::cout << "not enough cameras, skipping\n";
		return;
	}
	// calculate the pointcloud scale and 1st pointcloud
	else if (pgraph->cameras.size() == 2)
	{
		pointcloud_scale = pgraph->calculate_depthcloud_scale(pgraph->cameras.at(1), pgraph->cameras.at(0));

		std::shared_ptr<o3d::geometry::PointCloud> cloud = rgbd::create_pcloud_from_rgbd(*pgraph->cameras.at(0).rgb, 
			*pgraph->cameras.at(0).depth, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, false, pgraph->cameras.at(0).intr);

		cloud->Scale(pointcloud_scale, false);
		cloud->Transform(pgraph->cameras.at(0).T);
		pointclouds.emplace_back(cloud);
	}

	std::shared_ptr<o3d::geometry::PointCloud> cloud = rgbd::create_pcloud_from_rgbd(*cam.rgb, 
		*cam.depth, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, false, cam.intr);

	cloud->Scale(pointcloud_scale, false);
	cloud->Transform(cam.T);
	pointclouds.emplace_back(cloud);*/

	// remove overlap
	
	
}