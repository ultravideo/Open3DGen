#include "constants.h"

#include <cmath>

#include "math_modules.h"

/**
 *  NOTE:
 *      predict the acceleration of next frame
 *      predict the velocity of next frame
 *      predict the position of next frame
 *         
 *      generally:
 *          y = y0 + v0*t + 1/2*a*t^2
 * 
 *      derivation order:
 *      position -> velocity -> acceleration
 * 
 *  NOTE:
 *      confidence values based on temporal distance
 *      are calculated as (1 / (temporal difference))^(1/7)
 * 
 *      score values are calculated as max(ln(1/sqrt(position_diff)), ln(1/sqrt(angular_diff)))
 * 
 *  TODO:
 *      angular and linear acceleration
 */

namespace stitcher3d
{

class PosePredictor
{

public:
    PosePredictor();
    ~PosePredictor();

    /**
     *  adds a known good pose to the graph
     */
    void add_good_pose(const uint32_t frame_id, const Eigen::Vector3d position, const Eigen::Matrix3d rotation);

    /**
     *  predicts the next pose based on the previously input data.
     *  also returns the confidence value based on the time since last
     *  good pose, [0.0, 1.0]
     */
    std::tuple<Eigen::Vector3d, Eigen::Matrix3d, double> predict_next_pose(const uint32_t next_frame_id) const;

    /**
     *  verifies whether the given pose is valid.
     *  returns a score value based on how well the predicted and
     *  measured poses match (distance, velocity and angular)
     *  the confidence value based on 
     *  how much time has passed between frames. confidence ~ 1 / time.
     *  Both values are betwee n[0.0, 1.0]
     */
    std::tuple<double, double> verify_pose(const uint32_t frame_id, const Eigen::Vector3d position, const Eigen::Matrix3d rotation) const;

    Eigen::Vector3d velocity_from_to(const uint32_t begin, const uint32_t end) const;

    Eigen::Vector3d angular_velocity_from_to(const uint32_t begin, const uint32_t end) const;

private:

    inline double time_confidence(const uint32_t begin, const uint32_t end) const
    {
        // const double value = 1.0 / abs((double)end - (double)begin);
        // return pow(value, 1.0 / 7.0);

        /**
         *  TODO:
         *      rewrite the equations, not workign well
         */

        // const double score = log(1.0 / sqrt(abs((double)end - (double)begin)));
        // return std::min(std::max(score, 0.0), 1.0);

        return std::min(std::max(1.0 - ((double)end - (double)begin) * 0.1 + 0.1, 0.0), 1.0);
    }

    inline double pose_score(const double position_diff, const double angular_diff) const
    {
        // std::cout << "posdiff: " << position_diff << ", angular diff: " << math::rad2deg(angular_diff) << "\n";
        const double score = std::min(
            1.0 - sqrt(position_diff * 0.5),
            1.0 - sqrt(math::rad2deg(angular_diff) * 0.05)
        );

        return std::min(std::max(score, 0.0), 1.0);
    }

    uint32_t last_frame_index;

    struct PoseData
    {
        uint32_t frame_id;

        Eigen::Vector3d position;
        Eigen::Matrix3d rotation;

        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;

        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d angular_acceleration;
    };

    std::vector<PoseData> poses;

    PoseData next_preficted;

};


}