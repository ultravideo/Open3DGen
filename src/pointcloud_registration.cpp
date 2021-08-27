#include "pointcloud_registration.h"
#include "registration_params.h"

namespace stitcher3d
{
namespace registration
{

std::tuple<std::shared_ptr<surface::SurfaceMesh>, std::vector<std::shared_ptr<Camera>>, float> 
    register_pointcloud_sequence_features(const std::string& rgbd_path, const o3d::camera::PinholeCameraIntrinsic& intr,
    const std::vector<double>& distortion_coeffs, const RegistrationParams& rparams)
{
    std::vector<std::string> rgb_names;
    std::vector<std::string> d_names;
    {
        for (const auto& file : std::filesystem::directory_iterator(rgbd_path + RGB_PATH))
        {
            if (file.path().string().find(".png") == std::string::npos && file.path().string().find(".jpg") == std::string::npos)
                continue;

            rgb_names.push_back(file.path());
        }

        try
        {
            for (const auto& file : std::filesystem::directory_iterator(rgbd_path + DEPTH_PATH))
                d_names.push_back(file.path());
        } catch (...) { }

        std::sort(rgb_names.begin(), rgb_names.end(), [](const std::string& f, const std::string& s)
        {
            int f_i = std::stoi(utilities::split_string(f, '_').back());
            int s_i = std::stoi(utilities::split_string(s, '_').back());
            return f_i < s_i;
        });

        try
        {
            std::sort(d_names.begin(), d_names.end(), [](const std::string& f, const std::string& s)
            {
                int f_i = std::stoi(utilities::split_string(f, '_').back());
                int s_i = std::stoi(utilities::split_string(s, '_').back());
                return f_i < s_i;
            });
        } catch (...) { }
    }

    PoseGraph pgraph;
    pgraph.set_registration_parameters(rparams);
    // RealMesh rmesh = RealMesh(&pgraph);
    Timer t;
    std::shared_ptr<o3d::geometry::Image> depth = nullptr;

    std::vector<uint32_t> times;

    std::vector<int> position_ids;
    int consecutive_fail_count = 0;

    std::cout << "frames loaded, begin matching\n";

    for (int i = 0; i < rgb_names.size(); i++)
    {
        // skip according to stride
        if (i > 0 && i < rparams.skip_stride)
            continue;

        Timer a;
        const std::string& rgb_name = rgb_names.at(i);
        std::shared_ptr<o3d::geometry::Image> rgb = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
        o3d::io::ReadImage(rgb_name, *rgb);

        std::string depth_name = "";

        if (d_names.size() != 0)
        {
            depth_name = d_names.at(i);
            depth = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
            o3d::io::ReadImage(depth_name, *depth);
        }

        #ifdef DEBUG_MINIMAL
        std::cout << "\n\n" << rgb_name << "\n";
        #endif

        const std::string num_format = utilities::split_string(rgb_name, '_').back();
        const std::string num = utilities::split_string(num_format, '.')[0];
        const uint32_t time_id = std::stoi(num);

        if (!pgraph.add_camera_and_compute_pose_realtime(rgb, depth, intr, distortion_coeffs, time_id, rgb_name, depth_name))
        {
            #ifdef DEBUG_MINIMAL
            std::cout << "adding camera failed, continuing\n";
            #endif

            consecutive_fail_count++;
        }
        else
        {
            position_ids.push_back(i);
            consecutive_fail_count = 0;
        }

        if (consecutive_fail_count >= 30)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "terminated early, consequtive_fail_count >= 5\n";
            #endif

            break;       
        }

        times.push_back(a.lap_ms());
    }

    std::cout << "adding cameras finished: ";
    t.stop();


    // no need to re-triangulate if using Accurate -preset
    if (rparams.refine_cameras)
    {
        pgraph.retriangulate_landmarks();
        pgraph.recompute_camera_poses_landmark_PnP();
        t.stop("refined camera poses");

        // pgraph.bundle_adjust();
        // t.stop("bundle adjustment");
        // pgraph.retriangulate_landmarks();
    }
    
    pgraph.create_pointclouds_from_depths();
    // pgraph.landmarks_to_pointcloud();

    #ifdef DEBUG_VERBOSE
    uint32_t avg_time = 0;
    std::cout << "\n\n";
    for (auto ii : times)
    {
        avg_time += ii;
        std::cout << ii << ", ";
    }
    std::cout << "\n\n";

    std::cout << "average frametime: " << avg_time / times.size() << " ms\n";

    for (auto a : position_ids)
        std::cout << a << ", ";
    std::cout << "\n";
    #endif

    #ifdef DEBUG_VISUALIZE
    pgraph.visualize_tracks();
    #endif
    
    // std::exit(0);
    
    std::shared_ptr<surface::SurfaceMesh> mesh = std::make_shared<surface::SurfaceMesh>(surface::SurfaceMesh());
    mesh->pointclouds_from_posegraph(pgraph);
    // o3d::visualization::DrawGeometries({mesh->get_pointcloud()});
    // mesh->generate_mesh();
    // o3d::visualization::DrawGeometries({mesh->get_mesh()});

    return std::make_tuple(mesh, pgraph.get_cameras_copy(), pgraph.get_pointcloud_scale());
}

std::tuple<std::shared_ptr<surface::SurfaceMesh>,
    std::vector<o3d::geometry::Image>,
    std::vector<o3d::geometry::Image>,
    std::vector<Eigen::Matrix4d_u>> register_rgbd_sequence_for_surface(const std::string& rgbd_path,
                        const o3d::camera::PinholeCameraIntrinsic& intr, int frame_skip, int sequence_length)
{
    std::vector<o3d::geometry::Image> colors;
    std::vector<o3d::geometry::Image> depths;
    std::vector<Eigen::Matrix4d_u> camera_transforms;


    std::shared_ptr<surface::SurfaceMesh> mesh = 
        std::make_shared<surface::SurfaceMesh>(surface::SurfaceMesh());

    std::vector<std::string> rgb_names;
    std::vector<std::string> d_names;

    {
        for (const auto& file : std::filesystem::directory_iterator(rgbd_path + RGB_PATH))
            rgb_names.push_back(file.path());

        for (const auto& file : std::filesystem::directory_iterator(rgbd_path + DEPTH_PATH))
            d_names.push_back(file.path());

        std::sort(rgb_names.begin(), rgb_names.end(), [](const std::string& f, const std::string& s)
        {
            int f_i = std::stoi(utilities::split_string(f, '_').back());
            int s_i = std::stoi(utilities::split_string(s, '_').back());
            return f_i < s_i;
        });

        std::sort(d_names.begin(), d_names.end(), [](const std::string& f, const std::string& s)
        {
            int f_i = std::stoi(utilities::split_string(f, '_').back());
            int s_i = std::stoi(utilities::split_string(s, '_').back());
            return f_i < s_i;
        });
        
        if (rgb_names.size() != d_names.size())
            throw std::length_error("rgb and depth sizes do not match!");
    }

    const int end_index = sequence_length == -1 ? rgb_names.size() : frame_skip*sequence_length;

    std::shared_ptr<o3d::geometry::PointCloud> last_cloud;
    int failed_sequential = 0;

    for (int i = 0; i < end_index; i += frame_skip)
    {
        std::cout << "frame " << (int)(i / frame_skip) << " out of " << end_index / frame_skip << "\n";
        auto rgb = o3d::geometry::Image();
        o3d::io::ReadImage(rgb_names.at(i), rgb);
        auto d = o3d::geometry::Image();
        o3d::io::ReadImage(d_names.at(i), d);

        auto cloud = rgbd::create_pcloud_from_rgbd(rgb, d, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, true, intr);

        if (i == 0)
        {
            mesh->add_frame(cloud);
            last_cloud = cloud;
            camera_transforms.push_back(Eigen::Matrix4d_u::Identity());
            colors.push_back(rgb);
            depths.push_back(d);
            continue;
        }

        auto [transform, success] = stitcher3d::registration::register_pointclouds_slow(*cloud, *last_cloud, BEGIN_VOXEL_SIZE);
        
        if (!success)
        {
            std::cout << "stitch failed, continuing\n";
            failed_sequential++;
            if (failed_sequential >= STITCH_FAIL_ITER_COUNT)
            {
                std::cout << STITCH_FAIL_ITER_COUNT << " failed consequential stitches, aborting\n";
                break;
            }

            continue;
        }

        colors.push_back(rgb);
        depths.push_back(d);
        camera_transforms.push_back(transform);

        failed_sequential = 0;

        cloud = std::make_shared<o3d::geometry::PointCloud>(cloud->Transform(transform));
        mesh->add_frame(cloud);

        last_cloud = cloud;
        Visualizer::get_instance()->update(mesh->get_pointcloud());
    }

    mesh->post_process_pointcloud();
    Visualizer::get_instance()->update(mesh->get_pointcloud());
    
    std::cout << "pointcloud post processed\n";

    return std::make_tuple(mesh, colors, depths, camera_transforms);
}

std::tuple<Eigen::Matrix4d_u, bool> register_pointclouds_slow(const o3d::geometry::PointCloud& source, 
                                     const o3d::geometry::PointCloud& target,
                                     const float voxel_size, const float fitness_err_thr,
                                     const int max_iter_count)
{
    float current_voxel_size = voxel_size;

    std::vector<std::reference_wrapper<const o3d::registration::CorrespondenceChecker>> corr_checkers;
    o3d::registration::RegistrationResult result;
    double global_fitness = 0;


    // global registration
    for (int i = 0; i < max_iter_count; i++)
    {

        float distance_threshold = current_voxel_size / 2.0f;

        auto [s_down, s_fpfh] = downsample_and_fpfh_pcloud(source, current_voxel_size);
        auto [t_down, t_fpfh] = downsample_and_fpfh_pcloud(target, current_voxel_size);

        corr_checkers.clear();
        auto chk_edge_len = o3d::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        corr_checkers.push_back(chk_edge_len);
        auto chk_dist = o3d::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
        corr_checkers.push_back(chk_dist);
        auto chk_norm = o3d::registration::CorrespondenceCheckerBasedOnNormal(0.52359878);
        corr_checkers.push_back(chk_norm);

        result = o3d::registration::RegistrationRANSACBasedOnFeatureMatching(
            *s_down, *t_down, *s_fpfh, *t_fpfh, distance_threshold, o3d::registration::TransformationEstimationPointToPoint(false),
            4, corr_checkers, o3d::registration::RANSACConvergenceCriteria(RANSAC_MAX_ITER, RANSAC_MAX_VALIDATION)
        );

        global_fitness = result.fitness_;

        std::cout << "current iter " << i << " with voxel size of " << current_voxel_size << " and fitness of " << global_fitness << "\n";

        if (global_fitness > fitness_err_thr)
            break;

        current_voxel_size += voxel_size * ITER_VOXEL_SIZE_MULTIPLIER;

    }

    Eigen::Matrix4d_u transform = registration_ICP(result.transformation_, source, target, voxel_size);

    return std::make_tuple(transform, global_fitness > fitness_err_thr);
}

Eigen::Matrix4d_u colored_ICP(const Eigen::Matrix4d& initial_transform, const o3d::geometry::PointCloud& source,
    const o3d::geometry::PointCloud& target, const float voxel_size)
{
    std::vector<float> voxel_radius {voxel_size, voxel_size / 2.0f, voxel_size / 4.0f};
    std::vector<int> iters {50, 30, 14};
    Eigen::Matrix4d_u current_transform = initial_transform;

    for (int scale = 0; scale < 3; scale++)
    {
        int current_iter_count = iters.at(scale);
        float radius = voxel_radius.at(scale);

        auto source_down = source.VoxelDownSample(voxel_size);
        auto target_down = target.VoxelDownSample(voxel_size);

        source_down->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(radius * 2.0f, 30));
        target_down->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(radius * 2.0f, 30));

        auto result_icp = o3d::registration::RegistrationColoredICP(*source_down, *target_down, radius, current_transform,
            o3d::registration::ICPConvergenceCriteria(1e-6, 1e-6, current_iter_count));

        current_transform = result_icp.transformation_;
    }


    return current_transform;
}

Eigen::Matrix4d_u registration_ICP(const Eigen::Matrix4d& initial_transform, const o3d::geometry::PointCloud& source,
    const o3d::geometry::PointCloud& target, const float voxel_size)
{
    auto source_cpy = source.UniformDownSample(REFINE_DOWNSAMPLE);
    auto target_cpy = target.UniformDownSample(REFINE_DOWNSAMPLE);

    source_cpy->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
    target_cpy->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(0.1, 30));

    float distance_threshold = voxel_size * 0.4f;
    auto result = o3d::registration::RegistrationICP(
        *source_cpy, *target_cpy, distance_threshold, initial_transform, 
        o3d::registration::TransformationEstimationPointToPlane()
    );

    return result.transformation_;
}

down_fpfh downsample_and_fpfh_pcloud(const o3d::geometry::PointCloud& cloud, const float voxel_size)
{
    auto dcloud = cloud.VoxelDownSample(voxel_size);
    dcloud->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0f, 30));

    auto dfpfh = o3d::registration::ComputeFPFHFeature(*dcloud, 
            o3d::geometry::KDTreeSearchParamHybrid(voxel_size * 5.0f, 1000));

    return std::make_tuple(dcloud, dfpfh);
}


std::shared_ptr<surface::SurfaceMesh> register_rgbd_sequence(const std::string& rgbd_path,
                        const o3d::camera::PinholeCameraIntrinsic& intr, int frame_skip)
{
    std::shared_ptr<surface::SurfaceMesh> mesh = 
        std::make_shared<surface::SurfaceMesh>(surface::SurfaceMesh());

    std::vector<std::string> rgb_names;
    std::vector<std::string> d_names;

    for (const auto& file : std::filesystem::directory_iterator(rgbd_path + RGB_PATH))
        rgb_names.push_back(file.path());

    for (const auto& file : std::filesystem::directory_iterator(rgbd_path + DEPTH_PATH))
        d_names.push_back(file.path());

    std::sort(rgb_names.begin(), rgb_names.end(), [](const std::string& f, const std::string& s)
    {
        int f_i = std::stoi(utilities::split_string(f, '_').back());
        int s_i = std::stoi(utilities::split_string(s, '_').back());
        return f_i < s_i;
    });

    std::sort(d_names.begin(), d_names.end(), [](const std::string& f, const std::string& s)
    {
        int f_i = std::stoi(utilities::split_string(f, '_').back());
        int s_i = std::stoi(utilities::split_string(s, '_').back());
        return f_i < s_i;
    });
    
    if (rgb_names.size() != d_names.size())
        throw std::length_error("rgb and depth sizes do not match!");

    std::shared_ptr<o3d::geometry::PointCloud> last_cloud;
    int failed_sequential = 0;

    for (int i = 0; i < rgb_names.size(); i += frame_skip)
    {
        std::cout << "frame " << (int)(i / frame_skip) << " out of " << (int)(rgb_names.size() / frame_skip) << "\n";
        auto rgb = o3d::geometry::Image();
        o3d::io::ReadImage(rgb_names.at(i), rgb);
        auto d = o3d::geometry::Image();
        o3d::io::ReadImage(d_names.at(i), d);

        auto cloud = rgbd::create_pcloud_from_rgbd(rgb, d, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, true, intr);

        if (i == 0)
        {
            mesh->add_frame(cloud);
            last_cloud = cloud;
            continue;
        }


        auto [transform, success] = stitcher3d::registration::register_pointclouds_slow(*cloud, *last_cloud, BEGIN_VOXEL_SIZE);
        
        if (!success)
        {
            std::cout << "stitch failed, continuing\n";
            failed_sequential++;
            if (failed_sequential >= STITCH_FAIL_ITER_COUNT)
            {
                std::cout << STITCH_FAIL_ITER_COUNT << " failed consequential stitches, aborting\n";
                break;
            }

            continue;
        }

        failed_sequential = 0;

        cloud = std::make_shared<o3d::geometry::PointCloud>(cloud->Transform(transform));
        mesh->add_frame(cloud);

        last_cloud = cloud;
        Visualizer::get_instance()->update(mesh->get_pointcloud());

        // o3d::visualization::DrawGeometries({mesh->get_pointcloud()});
    }

    mesh->post_process_pointcloud();
    Visualizer::get_instance()->update(mesh->get_pointcloud());
    
    std::cout << "pointcloud post processed\n";
    return mesh;
}

}
}