#include "rgbd_utilities.h"


namespace stitcher3d
{
namespace rgbd
{


std::shared_ptr<o3d::geometry::PointCloud> create_pcloud_from_rgbd(
                        o3d::geometry::Image& rgb, o3d::geometry::Image& depth,
                        float depth_scale, float depth_clip_dist, bool invert,
                        const o3d::camera::PinholeCameraIntrinsic& intr)
{

    auto rgbd = o3d::geometry::RGBDImage::CreateFromColorAndDepth(rgb, depth, 
                                            1.f / depth_scale, depth_clip_dist, false);
    auto pointcloud = o3d::geometry::PointCloud::CreateFromRGBDImage(*(rgbd.get()), intr);

    // o3d::visualization::DrawGeometries({pointcloud});

    if (pointcloud->points_.size() == 0)
        return pointcloud;

    pointcloud = pointcloud->VoxelDownSample(DOWNSAMPLE_VOXEL_SIZE);

    std::vector<double> distances = pointcloud->ComputeNearestNeighborDistance();
    double avg_dist = std::accumulate(distances.begin(), distances.end(), (double)0) / (double)distances.size();
    auto [tempcloud, a] = pointcloud->RemoveRadiusOutliers(16, avg_dist*4.0f);
    pointcloud = tempcloud;

    pointcloud->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(0.1f, 16));
    pointcloud->OrientNormalsTowardsCameraLocation();

    if (invert)
        pointcloud->Transform(FLIP_TRANSFORM_4D);

    return pointcloud;
}

std::tuple<std::shared_ptr<o3d::geometry::PointCloud>, o3d::geometry::Image, o3d::geometry::Image>
        get_pointcloud_and_images(const std::string& rgb_filepath,
        const std::string& d_filepath,
        const o3d::camera::PinholeCameraIntrinsic& intr,
        bool invert)
{
    o3d::geometry::Image rgb, d;

    o3d::io::ReadImage(rgb_filepath, rgb);
    o3d::io::ReadImage(d_filepath, d);

    auto pcloud = create_pcloud_from_rgbd(rgb, d, DEFAULT_DEPTH_SCALE, 3.f, invert, intr);

    return std::make_tuple(pcloud, rgb, d);
}

}
}