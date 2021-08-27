#include "pose_predictor.h"

namespace stitcher3d
{

PosePredictor::PosePredictor()
{

}

PosePredictor::~PosePredictor()
{

}

void PosePredictor::add_good_pose(const uint32_t frame_id, const Eigen::Vector3d position, const Eigen::Matrix3d rotation)
{
    Eigen::Vector3d velocity, angular_velocity;
    if (poses.size() != 0)
    {
        angular_velocity = (math::eigen_rot2euler(rotation) - math::eigen_rot2euler(poses.back().rotation)) / 
            (abs((double)(frame_id - (double)poses.back().frame_id)));
        velocity = (position - poses.back().position) / 
            (abs((double)(frame_id - (double)poses.back().frame_id)));

        #ifdef DEBUG_VERBOSE
        std::cout << "angular velocity: " << angular_velocity.transpose() << "\n";
        std::cout << "velocity: " << velocity.transpose() << "\n";
        #endif
    }
    else
    {
        angular_velocity = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
    }


    const PoseData newpose
    {
        frame_id,

        position,
        rotation,

        // linear velocity and acceleration
        velocity,
        Eigen::Vector3d::Zero(),

        // angular velocity and acceleration
        angular_velocity,
        Eigen::Vector3d::Zero(),
    };

    poses.emplace_back(newpose);
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d, double> PosePredictor::predict_next_pose(const uint32_t next_frame_id) const
{
    if (poses.size() < 2)
        throw std::runtime_error("cannot predict a pose without reference frames!");

    const double time_diff = abs(((double)poses.back().frame_id - (double)next_frame_id));
    const double confidence = time_confidence(poses.back().frame_id, next_frame_id);

    // predict translation
    const Eigen::Vector3d t_pred = poses.back().position + 
        poses.back().velocity * time_diff +
        0.5 * poses.back().acceleration * time_diff * time_diff;

    // predict rotation

    const Eigen::Vector3d euler_pred = math::eigen_rot2euler(poses.back().rotation) + 
        poses.back().angular_velocity * time_diff + 
        0.5 * poses.back().angular_acceleration * time_diff * time_diff;

    const Eigen::Matrix3d r_pred = math::eigen_euler2rot(euler_pred);
    return std::make_tuple(t_pred, r_pred, confidence);
}

std::tuple<double, double> PosePredictor::verify_pose(const uint32_t frame_id, const Eigen::Vector3d position, const Eigen::Matrix3d rotation) const
{
    const auto [t_pred, r_pred, confidence] = predict_next_pose(frame_id);

    #ifdef DEBUG_VERBOSE
    std::cout << "position predicted: " << t_pred.transpose() << "\n";
    std::cout << "position measured: " << position.transpose() << "\n";

    std::cout << "rotation predicted: " << math::eigen_rot2euler(r_pred).transpose() << "\n";
    std::cout << "rotation measured: " << math::eigen_rot2euler(rotation).transpose() << "\n";
    #endif

    const double score = pose_score(
        (t_pred - position).norm(),
        (math::eigen_rot2euler(r_pred) - math::eigen_rot2euler(rotation)).norm());

    #ifdef DEBUG_VERBOSE
    std::cout << "pose score: " << score << ", confidence: " << confidence << ", temporal_diff: " << (int)frame_id - (int)poses.back().frame_id << "\n";
    std::cout << frame_id << ", " << poses.back().frame_id << "\n\n";
    #endif

    // if (poses.size() == 3)
    //     std::exit(0);

    return std::make_tuple(score, confidence);
}

Eigen::Vector3d PosePredictor::velocity_from_to(const uint32_t begin, const uint32_t end) const
{
    if (begin == end)
        throw std::runtime_error("begin and end cannotbe the same!");

    return (poses.at(end).position - poses.at(begin).position) / abs((double)end - (double)begin);
}


}