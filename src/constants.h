#pragma once

#include <Open3D/Open3D.h>
#include <cstdint>
#include <string>
#include "s3d_camera.h"

// either VERBOSE or MINIMAL, not both
#define DEBUG_VERBOSE
// #define DEBUG_MINIMAL

#define DEBUG_VISUALIZE

namespace stitcher3d
{

    // camera pose estimation parameters

    /**
     *  the ratio threshold of features, after which feature trail is terminated
     */
    static const float MATCH_RATIO_TRAIL_THRESHOLD = 0.25f;

    /**
     *  The minimum amount of features required for feature track
     */
    // static const uint32_t MATCH_COUNT_TRAIL_THRESHOLD = 200;

    static const uint32_t LANDMARK_MIN_COUNT = 30;
    // static const uint32_t RETRY_LANDMARK_COUNT = 25;

    /**
     *  Nearest neighbor matching ratio, by default 0.8 or 0.75
     */
    static const float NN_MATCH_RATIO = 0.75f;

    /**
     *  The maximum distance a feature match can have to be considered valid
     */
    static const float MAX_MATCH_DISTANCE = 200;
    static const int MIN_FEATURE_MATCH_COUNT = 10;

    // defaults to 0.001
    static const float AKAZE_THRESHOLD = 0.001f;
    static const float TRIANGULATED_POINT_OUTLIER_NORM = 100.0f;
    static const int FEATURE_2D_X_DIFF_PX = 60;
    static const float FEATURE_TRIANGULATION_MIN_DIFF = 30.0f;
    static const double MIN_CAMERA_TRANSLATION = 0.025;
    static const double MIN_CAMERA_ROTATION = 0.5;
    static const double MAX_CAMERA_TRANSLATION = 2.0;
    static const double MAX_CAMERA_ROTATION = 30.0;
    static const bool STITCH_ALL_FRAMES = false;
    static const float INPRECISE_FEATURE_THRESHOLD = 3.0f;
    static const float FEATURE_MATCHES_OUTLIER_RADIUS = 90.0f;
    static const float LANDMARK_FEATURE_MATCHES_OUTLIER_RADIUS = 60.0f;
    static const bool FEATURE_FILTER_CHECK_MIN = false;
    static const int CANDIDATE_EVERY_NTH = 7;

    static const double ADJUST_3DPOINT_NOISE = 0.4;
    static const double ADJUST_2DPOINT_NOISE = 2.0;

    static const double DEPTH_DISTANCE_OUTLIER = 500.0;

    // Distance threshold to identify inliers with homography check
    static const float INLIER_THRESHOLD = 4.5f;
    static const float CANDIDATE_INLIER_THRESHOLD = 3.5f;

    static const float BOUNDING_BOX_SCALE_FACTOR = 1.1;



    static const size_t CANDIDATE_CHAIN_LEN = 7;
    // after how many unsucceseful matches is the candidate
    // considered invalid
    static const size_t INVALID_CANDIDATE_CAMERA_TEMPORAL_DIFF = 5; // 15
    
    /**
     *  how many times greater than the average landmark 3d point's magnitude
     *  can the triangulated point's magnitude be
     */
    static const double LM_TRIANGULATION_DISTANCE_OUTLIER_MULTIPLIER = 3.0;


    static const int FIND_HOMOGRAPHY_RANSAC_ITERATIONS = 500;



    // how much movement is enough to triangulate into a lanrmark
    // relative to the pgraph's first 2 cameras translation
    static const double CANDIDATE_TRANSLATION_THRESHOLD = 0.45;

    // the magnitude of euler rotation, in degrees
    static const double CANDIDATE_ROTATION_THRESHOLD = 7.5;

    static const double CAMERA_POSITION_OUTLIER_MULTIPLIER = 5.0;
    static const double UV_PAD_SCALE = 0.85;

    // the % of diminished landmarks in view after which begin tracking
    // new candidates
    static const float NEW_CANDIDATES_THRESHOLD = 0.8f;

    // the % after which candidates will be made into new landmarks
    static const float NEW_LANDMARKS_FROM_CANDIDATES_THRESHOLD = 0.75f;

    // must be atleast 4
    static const size_t HOMOGRAPHY_MIN_MATCHES = 9;


    static const double POSE_CONFIDENCE_THRESHOLD = 0.6;
    static const double MIN_POSE_SCORE = 0.6;
    static const uint32_t POSE_PREDICTOR_CONSIDER_FRAME_COUNT = 50;
    
    static const double BAD_POSE_TRANSLATION_DIFF_MAGNITUDE = 0.5;
    static const double BAD_POSE_ANGULAR_DIFF_MAGNITUDE = 1.0;

    static const int SIFT_FEATURE_COUNT = 0;


    static const float UV_PRE_SCALAR = 0.008f;



    // other, partly deprecated constants

    namespace o3d = open3d;

    static const std::string DATAFILE_PATH(
        "data_files/");

    typedef std::tuple<std::shared_ptr<o3d::geometry::PointCloud>, std::shared_ptr<o3d::registration::Feature>> down_fpfh;

    static const std::vector<double> POCOF2_DISTORTION_COEFF_1280 { 0.06604662, -0.19640715, -0.00081006, 0.00031392, 0.20096299 };
    static const std::vector<double> POCOF2_DISTORTION_COEFF_1080 { 0.07125934, -0.1270845, 0.00058086, 0.0017399, 0.06769205 };
    static const std::vector<double> POCOF2_DISTORTION_COEFF_IVCAM_1080 { 6.61278222e-02, -1.91615174e-01, -1.21652015e-03, 9.69027934e-05, 2.28260836e-01 };

    // static const std::vector<double> REALSENSE_DISTORTION_COEFF_1280 { 1.63335313e-01, -4.65744623e-01, -2.56750047e-04, -1.82698621e-04, 3.18247809e-01 };
    static const std::vector<double> REALSENSE_DISTORTION_COEFF_1080 { 0.13206219, -0.30608398, -0.00084995, 0.00038177, 0.03328116 };
    // static const std::vector<double> REALSENSE_DISTORTION_COEFF_848 { 1.47788727e-01, -4.30653276e-01, -4.07154456e-04, 9.09620871e-06, 3.10979256e-01 };
    static const std::vector<double> REALSENSE_DISTORTION_COEFF_848 { 1.56923757e-01, -4.58242034e-01, -1.21292912e-03, -2.45781199e-04, 3.49355296e-01 };
    static const std::vector<double> REALSENSE_DISTORTION_COEFF_1280 { 1.66386586e-01, -5.26413220e-01, -1.01376611e-03, 1.59777094e-04, 4.65208008e-01 };

    static const std::vector<double> LOGI_WEBCAM_DISTORTION_COEFF_1280 { 0.06668624, -0.15275864, 0.00108533, -0.00024779,  0.00613592 };

    static const std::vector<double> LOGI_WEBCAM_DISTORTION_COEFF_800x600 { 0.07138038, -0.1977214, 0.00089949, -0.00028445, 0.07490953 };

    static const std::vector<double> ZERO_DISTORTION {0, 0, 0, 0, 0};

    static const std::vector<double> EVAL_CAM0_DISTORTION {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};

    static const Eigen::Matrix4d FLIP_TRANSFORM_4D = [] {
        Eigen::Matrix4d tmp;
        tmp <<  1, 0,  0,  0,
                0, -1, 0,  0,
                0, 0,  -1, 0,
                0, 0,  0,  1;
        return tmp;
    }();
    
    /**
     * @note do not touch, usede to flip realsense pointclouds!
     */
    static const Eigen::Matrix3d FLIP_TRANSFORM_3D = [] {
        Eigen::Matrix3d tmp;
        tmp << 1, 0, 0, 0, -1, 0, 0, 0, -1;
        return tmp;
    }();

    static const Eigen::Matrix3d FLIP_TRANSFORM_3D_SPECIAL = [] {
        Eigen::Matrix3d tmp;
        tmp << -1, 0, 0, 0, 1, 0, 0, 0, 1;
        return tmp;
    }();

    static const Eigen::Vector3d UP_VECTOR (0.f, 1.f, 0.f);

    static const Eigen::Vector3d RIGHT_VECTOR (1.f, 0.f, 0.f);

    static const Eigen::Vector3d FORWARD_VECTOR (0.f, 0.f, 1.f);

    static const Eigen::Vector2d UP_UVS (0.0f, 1.0f);
    static const Eigen::Vector2d RIGHT_UV (1.0f, 0.0f);

    static const Eigen::Vector3d ZERO_VECTOR_3D (0.0f, 0.0f, 0.0f);


    static const float DOWNSAMPLE_VOXEL_SIZE = 0.02f;

    static const std::string RGB_PATH = "rgb/";
    static const std::string DEPTH_PATH = "depth/";

    static const float STITCH_DISTANCE_MULTIPLIER = 1.5f;

    static const int STITCH_FAIL_ITER_COUNT = 5;

    static const float BEGIN_VOXEL_SIZE = 0.05f;

    static const float PP_VOXEL_SIZE = 0.01f;
    static const int PP_OUTLIER_NB_POINTS = 16;
    static const float PP_OUTLIER_MULTIPLIER = 3.5f;

    static const float DEFAULT_DEPTH_SCALE = 0.001f;
    static const float DEFAULT_CLOUD_FAR_CLIP = 3.65f;

    static const float ITER_VOXEL_SIZE_MULTIPLIER = 0.3f;

    static const int REFINE_DOWNSAMPLE = 4;


    static const int RANSAC_MAX_ITER = 4000000;
    static const int RANSAC_MAX_VALIDATION = 500;

    static const float UV_PADDING = 0.0003f;

    static const unsigned int TR_VERT_COUNT = 3;

    static const float UV_MAX_X = 1.0f;
    static const float UV_MIN_X = 0.0f;
    static const float UV_MAX_Y = 1.0f;
    static const float UV_MIN_Y = 0.0f;

    static const int POISSON_DEPTH = 9;
    static const float POISSON_SCALE = 1.1f;

    static const unsigned int LAPLACIAN_ITERATIONS = 1;
    static const float LAPLACIAN_LAMBDA = 0.5f;
    
    static const double PI = 3.14159265359;

    static const float FLOAT_SMALL = 0.000001f;
    static const float DOUBLE_SMALL = 0.000001f;

    static const std::string TEXTURE_PROJECTION_COMPUTE_SHADER_PATH = "texture_projection_compute.glsl";

    static const int GL_HITPOINT_OFFSET = 1;
    static const int GL_TRIANGLE_INDEX_SSBO_OFFSET = 2;
    static const int GL_OUT_COORD_OFFSET = 3;
    static const int GL_TRIANGLE_SSBO_OFFSET = 4;

    static const Eigen::Vector3d INVALID_HIT_POINT(0.f, 0.f, 0.f);

    static const std::string RAY_BLANK_IMG_8K = "blank_8k.png";
    static const std::string RAY_BLANK_IMG_4K = "blank_4k.png";
    static const std::string RAY_BLANK_IMG_2K = "blank_2k.png";

    static const uint8_t BLACK_THRESHOLD = 10;

    // in meters
    static const double POINT_MAX_BOUNDS_ABS = 6.0;

    static const float FLANN_FRATIO = 0.3f;

    static const int NOT_SET_CAM_ID = -1;

    static const float UNIT_SCALE = 1.0f;

    // static const Camera NULL_CAMERA {
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

    static const uint SCALE_ITER_COUNT = 10000;

    static const double SCALE_DIFF_THRESHOLD = 0.15;

    static const double MIN_DIST_SCALE_DIFFERENCE = 0.05;

    static const double MAX_FEATURE_ABS_DEVIATION = 250.0;

    static const std::map<uint, uint> WORK_GROUP_IMAGE_SIZE_LOOKUP {
        {1920*1080, 760},
        {1280*720, 900}
    };

    /**
     *  constants for OpenCV
     */


    // a sufficiently big number to represent infinity
    static const float LARGE_INFINITY = 10000.0;

    static const float ransac_thresh = 2.5f;

    static const std::string DEBUG_CAMERA_PATH = "debug_camera_mesh.obj";

    static const std::vector<Eigen::Vector3d> DEBUG_COLORS {
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(0.0, 1.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 1.0, 0.0),
        Eigen::Vector3d(0.5, 0.5, 0.5)
    };

    static const float POSE_SCALE_MIN_THRESHOLD = 0.2;

    static const float DEPTH_NEAR_CLIPPING = 0.05;

    /** @brief used to indicate the maximum allowed scale before throwing an error */
    static const float SCALE_FAILURE_UPPER = 20.0;

    static const uint MIN_TRACK_LENGTH = 3;

    static const Eigen::Vector3d CAMERA_FORWARD (0.0, 0.0, 1.0);

    static const float CAMERA_SCALE = 0.2f;

    static const uint32_t MIN_FEATURE_COUNT = 20;

    /**
     *  NOTE: the pose translation can be MAX_TRANL_DIFF_MULTIPLIER times 
     *      bigger than the previous pose translation
     */
    static const float MAX_TRANSL_DIFF_MULTIPLIER = 20.0f;

    static const double ABS_MAX_POSE_TRANSL = 1e6;

    /** 
     * Cameras must move at least MIN_TRANSLATION_RATIO * previous_translation
     */
    static const double MIN_TRANSLATION_RATIO = 0.20;
    static const double MAX_TRANSLATION_RATIO = 20.0;


    /**
     * Cameras must rotate at least MIN_RODRIQUES degrees
     */
    static const float MIN_RODRIQUES = 3.0f;
    static const float MAX_RODRIQUES = 25.0f;

    /**
     *  NOTE: the amount of pixels correspondences must have between
     *      matched features to be considered a new frame, makes sure
     *      pose scales are not insanely big.
     */    
    static const float MIN_CORRESPONDENSE_PIXEL_DIFF_INITIAL = 0.055f;
    static const float MIN_CORRESPONDENSE_PIXEL_DIFF = 0.02f; //0.04f;

    static const float MIN_FEATURE_DEVIATION_RATIO = 0.30f;

    static const float INITIAL_PIXEL_DIFF_RATIO = 0.30f;


    static const bool FORCE_INITIAL_NO_CHECK = false;

    // in pixels, in % 0.07 - 0.08

    static const float FEATURE_MATCHES_OUTLIER_RADIUS_RATIO =  0.1f; //0.085f;
    static const float FEATURE_MATCHES_OUTLIER_RADIUS_RATIO_MIN = 0.01f;
    
    // used to remove featurematches from, e.g. temporally stationary watermarks


    static const bool TERMINATE_NO_CORRESPONDENCE = true;


    static const float ICP_VOXEL_SIZE = 0.025f;
    static const float ICP_NORMAL_RADIUS = 0.1f;
    static const float ICP_FITNESS_THRESHOLD = 0.7f;
    static const uint32_t ICP_ITER_COUNT = 30;
    static const bool REFINE_ICP = false;

    static const uint32_t FORCE_ICP_AFTER_NO_CORRES_COUNT = 1;
    static const uint32_t FORCE_ICP_AFTER_NO_3VIEW_COUNT = 2;

    /**
     *  How many frames long must the loop closure be at least.
     *  Recommended to be quite long, because the delta between
     *  sequential frames is quite small and therefore a lot of 
     *  overlap is usually found.
     */
    static const uint32_t MIN_LOOP_LENGTH = 15;
    static const uint32_t MIN_LOOP_FEATURE_MATCH_COUNT = 15;
    static const bool SHOULD_LOOP_CLOSURE = false;

    static const double LOOP_3DPOINT_NOISE = 0.75;
    static const double LOOP_2DPOINT_NOISE = 4.0;

    typedef Eigen::Matrix<double, 3, 4> Mat34;

}

