#pragma once

// #include <Eigen/src/Core/Matrix.h>
#include <Open3D/Open3D.h>
#include <cstdint>
#include <fmt/core.h>
#include <iterator>
#include <sys/types.h>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <map>
#include <memory>
#include "constants.h"
#include "registration_params.h"
#include "s3d_camera.h"


namespace stitcher3d
{
namespace utilities
{

std::vector<std::string> split_string(const std::string& str, const char delim);


template <typename T>
std::vector<T> type_parse_str_vector(const std::vector<std::string>& str_vec);


inline std::tuple<std::string, std::string, std::string, uint32_t, uint32_t, uint32_t,
    std::vector<double>, uint32_t, uint32_t, std::string, std::string, bool,
    bool, float, bool, std::string, uint32_t, uint32_t, bool, double, uint32_t, 
    bool, float, float, registration::PipelineState> 
    parse_cmdline_args(int argc, char* argv[])
{
    std::string in_path;
    std::string intr_path;
    std::string out_path;
    uint32_t out_res = 4096;
    uint32_t out_nth = 1;
    uint32_t poisson_depth = POISSON_DEPTH;
    uint32_t cchain_len = CANDIDATE_CHAIN_LEN;
    uint32_t lam_min_count = LANDMARK_MIN_COUNT;
    std::vector<double> dist_coeffs = ZERO_DISTORTION;
    registration::PipelineState pstate = registration::PipelineState::Full;
    std::string state_str = "full";
    std::string serialized_cameras_path = "";
    std::string assets_path = DATAFILE_PATH;
    // registration::CameraPoseAlgorithm cam_pose_alg = registration::CameraPoseAlgorithm::Fast;
    bool refine_cameras = false;
    bool export_mesh = true;
    bool do_unwrap = true;
    bool write_interm_images = false;
    float simplify_voxel_size = 0.f;
    std::string load_mesh_path = "";
    uint32_t skip_stride = 0;
    uint32_t workgroup_size = 1024;
    double depth_far_clip = DEFAULT_CLOUD_FAR_CLIP;
    uint32_t max_feature_count = SIFT_FEATURE_COUNT;
    bool crop = true;
    float ray_threshold = 0.9;
    float reject_blur = 0.0;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg (argv[i]);

        if (arg == "in")
        {
            i++;
            in_path = std::string (argv[i]);
        }
        else if (arg == "reject_blur")
        {
            i++;
            reject_blur = std::stof(std::string(argv[i]));
        }
        else if (arg == "intr")
        {
            i++;
            intr_path = std::string(argv[i]);
        }
        else if (arg == "out")
        {
            i++;
            out_path = std::string (argv[i]);
        }
        else if (arg == "out_res")
        {
            i++;
            out_res = std::stoi(std::string(argv[i]));
        }
        else if (arg == "project_every_nth")
        {
            i++;
            out_nth = std::stoi(std::string(argv[i]));
        }
        else if (arg == "depth_far_clip")
        {
            i++;
            depth_far_clip = std::stod(std::string(argv[i]));
        }
        else if (arg == "max_feature_count")
        {
            i++;
            max_feature_count = std::stoi(std::string(argv[i]));
        }
        else if (arg == "poisson_depth")
        {
            i++;
            poisson_depth = std::stoi(std::string(argv[i]));
        }
        else if (arg == "write_interm_images")
        {
            i++;
            write_interm_images = std::string(argv[i]) == "true" ? true : false;
        }
        else if (arg == "crop")
        {
            i++;
            const std::string val (argv[i]);
            if (val == "true")
                crop = true;
            else
                crop = false;
        }
        else if (arg == "unwrap")
        {
            i++;
            if (std::string(argv[i]) == "true")
                do_unwrap = true;
            else if (std::string(argv[i]) == "false")
                do_unwrap = false;
            else
            {
                std::cout << "invalid option for unwrap: " << std::string(argv[i]) << ", aborting\n";
                std::exit(0);
            }
        }
        else if (arg == "simplify_voxel_size")
        {
            i++;
            simplify_voxel_size = std::stof(std::string(argv[i]));
        }
        else if (arg == "load_mesh_path")
        {
            i++;
            load_mesh_path = std::string(argv[i]);
        }
        else if (arg == "dist_coeff")
        {
            i++;
            const std::string dcpath (argv[i]);
            std::ifstream file (dcpath);
            dist_coeffs.clear();

            if (file.is_open())
            {
                std::string line;
                getline(file, line);
                std::vector<std::string> split_coeffs = utilities::split_string(line, ' ');
                for (const auto& s : split_coeffs)
                    dist_coeffs.push_back(std::stod(s));
            }
            else
            {
                std::cout << "cannot open file " << dcpath << ", aborting\n";
                std::exit(0);
            }
            file.close();
        }
        else if (arg == "cchain_len")
        {
            i++;
            cchain_len = std::stoi(std::string(argv[i]));
        }
        else if (arg == "lam_min_count")
        {
            i++;
            lam_min_count = std::stoi(std::string(argv[i]));
        }
        else if (arg == "workgroup_size")
        {
            i++;
            workgroup_size = std::stoi(std::string(argv[i]));
        }
        else if (arg == "ser_cameras_cloud")
        {
            i++;
            serialized_cameras_path = std::string(argv[i]);
        }
        else if (arg == "assets_path")
        {
            i++;
            assets_path = std::string(argv[i]);
        }
        else if (arg == "skip_stride")
        {
            i++;
            skip_stride = std::stoi(std::string(argv[i]));
        }
        else if (arg == "ray_threshold")
        {
            i++;
            ray_threshold = std::stof(std::string(argv[i]));
        }
        else if (arg == "export_mesh_interm")
        {
            i++;
            if (std::string(argv[i]) == "true")
                export_mesh = true;
            else if (std::string(argv[i]) == "false")
                export_mesh = false;
            else
            {
                std::cout << "invalid export mesh value " << std::string(argv[i]) << ", aborting\n";
                std::exit(0);
            }
        }
        else if (arg == "refine_cameras")
        {
            i++;
            const std::string alg_str (argv[i]);

            if (alg_str == "true")
                refine_cameras = true;
            else if (alg_str == "false")
                refine_cameras = false;
            else
            {
                std::cout << "invalid refine_cameras argument " << refine_cameras << ", aborting\n";
                std::exit(0);
            }
        }
        else if (arg == "pipeline")
        {
            i++;
            state_str = std::string (argv[i]);

            if (state_str == "full")
                pstate = registration::PipelineState::Full;
            else if (state_str == "only_pointcloud")
                pstate = registration::PipelineState::PointcloudOnly;
            else if (state_str == "only_mesh")
                pstate = registration::PipelineState::MeshOnly;
            else if (state_str == "only_project")
                pstate = registration::PipelineState::ProjectOnly;
            else
            {
                std::cout << "invalid pipeline state option: '" << arg << "', aborting\n";
                std::exit(0);
            }

        }
        else
        {
            std::cout << "invalid argument: '" << arg << "', aborting\n";
            std::exit(0);
        }
    }

    std::cout << "reconstruction will be done using the following parameters:\n"
        << "   input image sequence path: " << in_path << "\n"
        << "   camera intrinsics: " << intr_path << "\n"
        << "   output path: " << out_path << "\n"
        << "   output texture resolution: " << out_res << "\n"
        << "   every " << out_nth << " frame will contribute to texture projection\n"
        << "   with poisson depth of " << poisson_depth << "\n"
        << "   with distortion coefficients ";

        for (auto d : dist_coeffs)
            std::cout << d << " ";

        std::cout <<"\n"
        << "   with feature candidate chain length of " << cchain_len << "\n"
        << "   serialized cameras and pointcloud path: " << serialized_cameras_path << "\n"
        << "   camera pose algorithm: " << refine_cameras << "\n"
        << "   load mesh from: " << load_mesh_path << "\n"
        << "   export intermediary mesh: " << export_mesh << "\n"
        << "   load assets from: " << assets_path << "\n"
        << "   depth_far_clip of " << depth_far_clip << "\n"
        << "   write per-frame images: " << write_interm_images << "\n"
        << "   the mesh will be decimated to " << simplify_voxel_size << " voxels (0 is unlimited)\n"
        << "   with projection compute shader workgroup size of " << workgroup_size << "\n"
        << "   will the mesh be UV unwrapped: " << do_unwrap << "\n"
        << "   a minimum feature-landmark for succesful frame: " << lam_min_count << "\n"
        << "   images with blur value of over " << reject_blur << " will be rejected in texture projection\n"
        << "   " << max_feature_count << " features will be detected\n"
        << "   " << state_str << " will be constructed\n"
        << "   with ray-normal dot product threshold of " << ray_threshold << "\n"
        << "   will the mesh be cropped: " << crop <<  "\n";

    return std::make_tuple(in_path, intr_path, out_path, out_res, out_nth, 
        poisson_depth, dist_coeffs, cchain_len, lam_min_count, 
        serialized_cameras_path, assets_path, refine_cameras, export_mesh,
        simplify_voxel_size, do_unwrap, load_mesh_path, skip_stride, workgroup_size, 
        write_interm_images, depth_far_clip, max_feature_count, crop, ray_threshold, reject_blur, pstate);
}

template <class K, class V>
inline typename std::map<V, K>::iterator map_find_by_value(const V val, std::map<V, K>& data)
{
    for (typename std::map<V, K>::iterator it = data.begin(); it != data.end(); it++)
    {
        if (it->second == val)
            return it;
    }

    return data.end();
}

template <class K, class V>
inline std::vector<K> get_keys_as_vec(std::map<K, V>& data)
{
    std::vector<K> keys;
    keys.reserve(data.size());

    for (auto& d : data)
        keys.push_back(d.first);

    return keys;
}

template <class K, class V>
inline std::vector<V> get_column_values_at_as_vec(const size_t index, std::map<K, std::vector<V>>& data)
{
    std::vector<V> column;
    column.reserve(data.size());

    for (auto& d : data)
        column.push_back(d.second[index]);

    return column;
}

o3d::camera::PinholeCameraIntrinsic read_camera_intrinsics(const std::string& filename);

inline Camera deserialize_cam_from_str(const std::string& datastr)
{
    const std::vector<std::string> elems = split_string(datastr, ';');

    const uint32_t id = std::stoi(elems[0]);
    const std::string rgb_path = elems[1];
    const std::string depth_path = elems[2];
    const std::string Tstr = elems[3];

    Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
    int i = 0;
    for (const std::string& s : split_string(Tstr, ','))
    {
        T(i) = std::stod(s);
        i++;
    }

    //  Camera NULL_CAMERA {
    //     Eigen::Vector3d::Zero(),
    //     Eigen::Matrix3d::Identity(),
    //     UNIT_SCALE,
    //     Eigen::Vector3d::Zero(),
    //     Eigen::Matrix3d::Identity(),
    //     Eigen::Matrix4d::Identity(),
    //     Eigen::Matrix<double, 3, 4>::Identity(),
    //     ZERO_DISTORTION,
    //     nullptr,
    //     std::vector<cv::KeyPoint>(),
    //     cv::Mat(),
    //     nullptr,
    //     nullptr,
    //     nullptr,
    //     nullptr,
    //     o3d::camera::PinholeCameraIntrinsic(),
    //     -1,
    //     0.0f,
    //     "",
    //     ""
    // };
    
    Camera cam;
    {
        cam.position = Eigen::Vector3d::Zero();
        cam.rotation = Eigen::Matrix3d::Identity();
        cam.scale = 0.0;
    }

    cam.id = id;
    cam.rgb_name = rgb_path;
    cam.depth_name = depth_path;
    cam.T = T;

    auto depth = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
    o3d::io::ReadImage(depth_path, *depth);
    cam.depth = depth;

    auto rgb = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
    o3d::io::ReadImage(rgb_path, *rgb);
    cam.rgb = rgb;

    return cam;
}

inline std::string serialize_cameras_to_str(const std::vector<std::shared_ptr<Camera>>& cams, const double pcloud_scale)
{
    std::string cam_serialized = "";

    for (const std::shared_ptr<Camera> cam : cams)
    {
        const std::string depth_name = cam->depth_name;
        const std::string rgb_name = cam->rgb_name;
        const Eigen::Matrix4d T = cam->T;
        const uint32_t id = cam->id;

        std::string Tstr = "";
        for (int i = 0; i < 16; i++)
        {
            Tstr += std::to_string(T(i)) + ",";
        }

        cam_serialized += std::to_string(id) + ";" + rgb_name + ";" + depth_name + ";" + Tstr + "\n";
    }

    cam_serialized += std::to_string(pcloud_scale) + "\n";

    return cam_serialized;
}

inline void replace_string(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return;
        
    str.replace(start_pos, from.length(), to);
}

}
}