#include "poseg2.h"
#include "features.h"
#include "math_modules.h"
#include "utilities.h"
#include <Eigen/src/Core/Matrix.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Visualization/Utility/DrawGeometry.h>
#include <boost/core/use_default.hpp>
#include <cstdint>
#include <functional>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

/**
 *  TODO:
 *      - cache landmark descriptors, update only when added to landmark
 *      - compute pointcloud scale as soon as possible
 *          --> can use metric units
 */

namespace stitcher3d
{
    PoseGraph::PoseGraph() :
        cameras(std::vector<Camera*>()),
        // f_matcher(std::make_unique<cv::BFMatcher>(cv::BFMatcher(cv::NORM_HAMMING)))
        // NORM_HAMMING for ORB, AKAZE. NORM_L2 for SIFT
        f_matcher_crosscheck(std::make_unique<cv::BFMatcher>(*cv::BFMatcher::create(cv::NORM_L2, true))),
        f_matcher(std::make_unique<cv::BFMatcher>(*cv::BFMatcher::create(cv::NORM_L2, false)))
    { }

    PoseGraph::~PoseGraph()
    {
        for (Camera* cam : cameras)
            delete cam;
    }

    Camera* PoseGraph::add_camera(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation,
        const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& depth,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const std::string rgb_name, const std::string depth_name, const bool offline)
    {
        // const auto [cv_keypoints, descriptor] = features::detect_akaze_features(rgb, &intr, &distortion_coeffs);
        Camera* tcam;

        auto cv_rgb = nullptr; //std::make_shared<cv::Mat>(rgbd::o3d_image_to_cv_image(rgb));
        auto cv_depth = nullptr; //std::make_shared<cv::Mat>(rgbd::o3d_image_to_cv_image(depth, true));

        // cv::imshow("depth", *cv_depth);
        // cv::waitKey(0);

        if (!offline)
        {
            tcam = new Camera (
                position,
                rotation,
                UNIT_SCALE,
                position,
                rotation,
                Eigen::Matrix4d::Identity(),
                intr.intrinsic_matrix_ * Eigen::Matrix<double, 3, 4>::Identity(),
                distortion_coeffs,
                nullptr,
                std::vector<cv::KeyPoint>(),
                cv::Mat(),
                rgb,
                depth,
                cv_rgb,
                cv_depth,
                intr,
                NOT_SET_CAM_ID,
                0.0f,
                rgb_name,
                depth_name
            );
        }
        else
        {
            throw std::runtime_error("unsupported feature, aborting");
        }
        
        // dR.transpose(), -dR.transpose() * dt
        // set_camera_T_and_P(tcam, Eigen::Matrix3d::Identity().transpose(), -Eigen::Matrix3d::Identity().transpose() * Eigen::Vector3d::Zero());

        cameras.push_back(tcam);
        cameras.back()->id = cameras.size() - 1;
        return cameras.back();
    }

    void PoseGraph::add_camera_offline(const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& depth,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const std::string rgb_name, const std::string depth_name)
    {
        Camera* cam = add_camera(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 
            rgb, depth, intr, distortion_coeffs, rgb_name, depth_name, true);

        const auto [cv_keypoints, descriptor] = features::detect_features(rgb, reg_params.max_feature_count, &intr, &distortion_coeffs);
        cam->kp_features = cv_keypoints;
        cam->feature_descriptors = descriptor;
    }

    bool PoseGraph::add_camera_and_compute_pose_realtime(const std::shared_ptr<o3d::geometry::Image>& rgb, const std::shared_ptr<o3d::geometry::Image>& depth,
        const o3d::camera::PinholeCameraIntrinsic& intr, const std::vector<double>& distortion_coeffs,
        const uint32_t time_id, const std::string rgb_name, const std::string depth_name)
    {
        // if (landmarks.size() > 200000)
        //     visualize_tracks();

        Timer t;

        #ifdef DEBUG_VERBOSE
        std::cout << "\n";
        #endif

        Camera* cam = add_camera(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), rgb, depth, intr, distortion_coeffs, rgb_name, depth_name);

        const auto [cv_keypoints, descriptor] = features::detect_features(rgb, reg_params.max_feature_count, &intr, &distortion_coeffs);
        cam->kp_features = cv_keypoints;
        cam->feature_descriptors = descriptor;

        #ifdef DEBUG_VERBOSE
        t.stop("feature detection");
        #endif

        if (cameras.size() == 2)
        {
            #ifdef DEBUG_VERBOSE
        	std::cout << "less than 3 cameras, using essential matrix\n";
            #endif

        	const bool status = compute_initial_pose(cam, cameras.at(0));

            #ifdef DEBUG_VERBOSE
        	t.stop("computing initial pose");
            #endif

            if (status)
                pose_predictor.add_good_pose(time_id, cameras.back()->position, cameras.back()->rotation);

        	return status;
        }
        else if (cameras.size() == 1)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "only one camera, returning\n";
            #endif

            pose_predictor.add_good_pose(time_id, cam->position, cam->rotation);
            return true;
        }
        // visualize_tracks();

        const bool status = compute_pose_realtime(cam, time_id);

        #ifdef DEBUG_VERBOSE
        t.stop("computing the entire pose");
        #endif

        if (!status)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "adding camera failed, removing\n";
            #endif

            remove_last_camera(cameras.back());
        }

        return status;
    }

    void PoseGraph::compute_camera_poses_accurate()
    {
        if (cameras.size() < 2)
            throw std::runtime_error("not enough cameras to compute poses, aborting");

        Timer t;

        // pair-wise match all cameras
        for (int ii = 0; ii < cameras.size(); ii++)
        {
            OfflineCamera* refcam = dynamic_cast<OfflineCamera*>(cameras.at(ii));

            for (int jj = ii + 1; jj < cameras.size(); jj++)
            {
                OfflineCamera* cam = dynamic_cast<OfflineCamera*>(cameras.at(jj));

                const std::vector<std::pair<uint32_t, uint32_t>> rc_corr = 
                    match_features_hgraphy(refcam->feature_descriptors, cam->feature_descriptors, refcam, cam);

                    // DEBUG_visualize_matches(refcam, cam, rc_corr);

                // loop through matches and add to Camera match-map for each feature index (fi)
                for (int fi = 0; fi < rc_corr.size(); fi++)
                {
                    // add feature chains
                    refcam->kp_match_index_map[rc_corr[fi].first][jj] = rc_corr[fi].second;
                    cam->kp_match_index_map[rc_corr[fi].second][ii] = rc_corr[fi].first;
                }
            }
        }

        #ifdef DEBUG_VERBOSE
        t.stop("pair-wise correspondences and feature matches");
        #endif

        /**
         *  TODO:
         *      - find corresponding candidates in each image, pair-wise -> map
         *      - loop through all cameras, for each: loop through all matched frames
         *      - after local pose recovery, triangulate points
         *      - scale points 
         *      - global pose recovery
         *      - triangulate good points
         *      - average landmarks
         *          - fix camera poses with PnP?
         * 
         *  ALGORITHM:
         *      - pairwise match features and add to corresponding frame lookups
         *      - fix 0th camera as origin
         *      - iterate cameras:
         *          - recover pose
         *          - when featurecount < threshold (0.5?), stop and set as unit transformation
         *          - this is makes a fragment, a camera sequence
         *      - refcam +1 and repeat
         *      - match fragments
         *          - 2nd of first, 1st of second are the same frame
         *          - set 1st of 2nd transformation
         *          - propagate transformations (relative to 1st of 2nd)
         */

        std::vector<FrameFragment> frame_fragments;


        cv::Mat intr_cv;
        cv::eigen2cv(cameras.at(0)->intr.intrinsic_matrix_, intr_cv);

        /**
         *  ii is the fragment's 1st camera
         *  jj is the fragment's last camera
         */
        for (int ii = 0; ii < cameras.size() - 1; ii++)
        {
            OfflineCamera* refcam = dynamic_cast<OfflineCamera*>(cameras.at(ii));

            // set the refcam to be at origin, dealing with relative fragments
            set_camera_T_and_P(refcam, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

            const uint32_t next_cam_id = refcam->id + 1;
            
            // <feature index, indexes in consequent frames>
            // feature must be present in all of them, otherwise delete
            std::map<uint32_t, std::vector<uint32_t>> feature_tracks;

            // for (int i = 0; i < refcam->kp_match_index_map.size(); i++)

            // construct feature trails

            // populate reference feature id lookup
            for (auto a : refcam->kp_match_index_map)
            {
                if (a.second.count(next_cam_id) != 0)
                {
                    feature_tracks[a.first] = { a.second.at(next_cam_id) };
                }
            }

            uint32_t next_cam_loop_id = next_cam_id;
            std::map<uint32_t, std::map<uint32_t, uint32_t>>& frame_kp_matches = dynamic_cast<OfflineCamera*>(cameras.at(next_cam_loop_id))->kp_match_index_map;
            uint32_t current_match_count = feature_tracks.size();
            const uint32_t initial_match_count = feature_tracks.size();

            while ((float)current_match_count > (float)initial_match_count * MATCH_RATIO_TRAIL_THRESHOLD )
                // || current_match_count > MATCH_COUNT_TRAIL_THRESHOLD)
            {
                std::vector<uint32_t> to_remove_tracks;

                for (auto& track : feature_tracks)
                {
                    const uint32_t ref_f_id = track.first;
                    const uint32_t fid = track.second.back();

                    // check if feature f is found also in the next_cam's matches
                    if (frame_kp_matches.at(fid).count(next_cam_loop_id + 1) != 0)
                    {
                        track.second.push_back(frame_kp_matches[fid][next_cam_loop_id + 1]);
                    }
                    // feature doesn't exist in next, mark for delete
                    else
                        to_remove_tracks.push_back(ref_f_id);
                }

                // remove tracks which weren't long enough 
                for (uint32_t to_remove_id : to_remove_tracks)
                    feature_tracks.erase(to_remove_id);

                current_match_count = feature_tracks.size();

                // NOTE: throw away the last frame, should be fixed at some point
                // if (next_cam_loop_id >= cameras.size() - 1)
                //     break;

                next_cam_loop_id++;
                frame_kp_matches = dynamic_cast<OfflineCamera*>(cameras.at(next_cam_loop_id))->kp_match_index_map;

                // std::cout << "next camera: " << next_cam_loop_id << ", track size: " 
                //     << current_match_count << ", out of " << initial_match_count << "\n";
            }

            std::cout << "feature trail feature count: " << feature_tracks.size() << 
                " and length: " << (feature_tracks.size() > 0 ? feature_tracks.begin()->second.size() : 0) << "\n";

            // for (auto a : feature_tracks)
            // {
            //     for (auto b : a.second)
            //         std::cout << b << " ";
            //     std::cout << "\n";
            // }

            // recover relative poses between frames
            // for (int jj = ii + 1; jj < cameras.size(); jj++)

            OfflineCamera* cam = dynamic_cast<OfflineCamera*>(cameras.at(ii + feature_tracks.begin()->second.size()));

            const std::vector<uint32_t> refcam_track_features = utilities::get_keys_as_vec(feature_tracks);
            const std::vector<uint32_t> cam_track_features = utilities::get_column_values_at_as_vec(feature_tracks.begin()->second.size() - 1, feature_tracks);

            /**
             *  - get 2d cv points 
             *  - recover poses for first and last
             *  - triangulate
             *  - PnP for mid frames
             */

            std::vector<cv::Point2f> ref_points, cam_points;
            ref_points.reserve(refcam_track_features.size());
            cam_points.reserve(cam_track_features.size());

            std::vector<std::pair<uint32_t, uint32_t>> fmatches;

            for (int i = 0; i < refcam_track_features.size(); i++)
            {
                const cv::Point2f ref_pt = refcam->kp_features[refcam_track_features[i]].pt;
                const cv::Point2f cam_pt = cam->kp_features[cam_track_features[i]].pt;

                fmatches.push_back(std::make_pair(refcam_track_features[i], cam_track_features[i]));

                ref_points.push_back(ref_pt);
                cam_points.push_back(cam_pt);

                // std::cout << ref_pt << " : " << cam_pt << "\n";
            }

            cv::Mat good_feature_mask;

            const cv::Mat essential = cv::findEssentialMat(ref_points, cam_points, intr_cv, cv::RANSAC, 0.999, 1.0, good_feature_mask);
            cv::Mat R_local, t_local;
            cv::recoverPose(essential, ref_points, cam_points, intr_cv, R_local, t_local, good_feature_mask);

            Eigen::Matrix3d dR;
            Eigen::Vector3d dt;
            cv::cv2eigen(R_local, dR);
            cv::cv2eigen(t_local, dt);

            cam->dR = dR;
            cam->dt = dt;

            // DEBUG_visualize_matches(refcam, cam, fmatches);

            set_camera_T_and_P(cam, dR.transpose(), -dR.transpose() * dt);
            set_camera_T_and_P(cam, refcam->T * cam->T);

            // std::cout << cam->rotation << "\n\n" << cam->position.transpose() << "\n\n";

            cv::Mat refcam_P, cam_P;
            cv::eigen2cv(refcam->P, refcam_P);
            cv::eigen2cv(cam->P, cam_P);

            cv::Mat p4d;
            cv::triangulatePoints(refcam_P, cam_P, ref_points, cam_points, p4d);

            std::vector<Eigen::Vector3d> points3d;

            std::vector<cv::Point3f> new_points, existing_points;

            // don't scale the points if no landmarks are found
            if (landmarks.size() != 0)
            {
                for (int jj = 0; jj < ref_points.size(); jj++)
                {
                    const cv::Point3f p3d (
                        p4d.at<float>(0, jj) / p4d.at<float>(3, jj),
                        p4d.at<float>(1, jj) / p4d.at<float>(3, jj),
                        p4d.at<float>(2, jj) / p4d.at<float>(3, jj));


                    if (good_feature_mask.at<uint32_t>(jj))
                    {
                        // points3d.push_back(Eigen::Vector3d(p3d.x, p3d.y, p3d.z));

                        // landmark

                        const uint32_t ref_f_id = refcam_track_features[jj];
                        // const uint32_t cam_f_id = cam_track_features[jj];

                        // check if landmark exists -> add to / create new
                        if (refcam->landmark_kp_exists(ref_f_id) && refcam->match_exists(ref_f_id, ii))
                        {
                            const uint32_t lm_id = refcam->landmark_lookup[ref_f_id];
                            const cv::Point3f avg_3d = landmarks[lm_id].sum_point / (float)(landmarks[lm_id].view_data.size() - 1);

                            // save the points for scale calculations
                            new_points.push_back(p3d);
                            existing_points.push_back(avg_3d);
                        }
                    }
                }

                // calculate the scale

                double scale = 0.0;
                uint32_t count = 0;

                for (uint32_t jj = 0; jj < new_points.size() - 1; jj++)
                {
                    for (uint32_t kk = jj + 1; kk < new_points.size(); kk++)
                    {
                        const double denominator = norm(new_points[jj] - new_points[kk]);
                        if (math::is_close(denominator, 0.f))
                            continue;

                        const double s = norm(existing_points[jj] - existing_points[kk]) / denominator;

                        scale += s;
                        count++;
                    }
                }

                assert(count);

                scale /= (double)count;
                std::cout << "for image " << ii << " final scale is " << scale << "\n";

                dt *= scale;

                cam->dR = dR;
                cam->dt = dt;

                // DEBUG_visualize_matches(refcam, cam, fmatches);

                set_camera_T_and_P(cam, dR.transpose(), -dR.transpose() * dt);
                set_camera_T_and_P(cam, refcam->T * cam->T);
                cv::eigen2cv(cam->P, cam_P);

                // re-triangulate with new projection matrix
                cv::triangulatePoints(refcam_P, cam_P, ref_points, cam_points, p4d);
            }

            std::vector<cv::Point3f> tr_3d_points;

            for (int jj = 0; jj < ref_points.size(); jj++)
            {
                const cv::Point3f p3d (
                    p4d.at<float>(0, jj) / p4d.at<float>(3, jj),
                    p4d.at<float>(1, jj) / p4d.at<float>(3, jj),
                    p4d.at<float>(2, jj) / p4d.at<float>(3, jj));

                // add to or create new landmarks
                if (good_feature_mask.at<uint32_t>(jj))
                {
                    points3d.push_back(Eigen::Vector3d(p3d.x, p3d.y, p3d.z));
                    tr_3d_points.push_back(p3d);

                    const uint32_t ref_f_id = refcam_track_features[jj];
                    const uint32_t cam_f_id = cam_track_features[jj];

                    // landmark exists, add to it
                    if (refcam->landmark_kp_exists(ref_f_id))
                    {
                        cam->landmark_lookup[cam_f_id] = refcam->landmark_lookup[ref_f_id];

                        Landmark& lm = landmarks[cam->landmark_lookup[cam_f_id]];
                        lm.add_view_simple(cam->id, cam_f_id, p3d);
                    }
                    // landmark doesn't exist, create new
                    else
                    {
                        Landmark lm;
                        lm.add_view_simple(refcam->id, ref_f_id, p3d);
                        lm.add_view_simple(cam->id, cam_f_id, p3d);

                        landmarks.emplace_back(lm);

                        refcam->landmark_lookup[ref_f_id] = landmarks.size() - 1;
                        cam->landmark_lookup[cam_f_id] = landmarks.size() - 1;
                    }
                }
            }


            std::shared_ptr<o3d::geometry::TriangleMesh> camera_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            o3d::io::ReadTriangleMeshFromOBJ(reg_params.assets_path + DEBUG_CAMERA_PATH, *camera_mesh, false);
            camera_mesh->Transform(cam->T);

            auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(points3d));
            o3d::visualization::DrawGeometries({ cloud, camera_mesh });

            // create frame fragment, lambdas ftw
            frame_fragments.push_back(FrameFragment {
                [&]{ std::vector<Camera*> cams; for (int i = ii; i < ii + feature_tracks.begin()->second.size(); i++) cams.push_back(cameras[i]); return cams; } (),
                feature_tracks,
                [&] { std::vector<Eigen::Matrix4d> ts; for (int i = ii; i < ii + feature_tracks.begin()->second.size(); i++) ts.push_back(cameras[i]->T); return ts; }(),
                tr_3d_points
            });
        }

        // PnP for first fragment
        FrameFragment& initial = frame_fragments[0];
        for (int ii = 1; ii < initial.frag_cameras.size() - 2; ii++)
        {
            // take 2d feature points, PnP to initial.tr_3d_points
            const auto f_ids = utilities::get_column_values_at_as_vec(ii, initial.frag_feature_tracks);

            Camera* cam = initial.frag_cameras[ii];

            std::vector<cv::Point2f> view_features_points;

            for (int jj = 0; jj < f_ids.size(); jj++)
                view_features_points.push_back(cam->kp_features[f_ids[jj]].pt);

            const cv::Mat cv_intr = [&]{
                cv::Mat intr;
                cv::eigen2cv(cam->intr.intrinsic_matrix_, intr);
                return intr;
            }();

            cv::Mat rcv, tcv;
            cv::solvePnPRansac(initial.feature_3d_points, view_features_points, cv_intr, cam->distortion_coeffs, rcv, tcv, true, 1000, 4.0, 0.987);

            Eigen::Matrix4d R;
            Eigen::Vector3d t;

            cv::cv2eigen(rcv, R);
            cv::cv2eigen(tcv, t);

            // NOTE: this doesn't work anymore!
            assert(false);
            // set_camera_T_and_P(cam, R.transpose(), -R.transpose() * t);
        }

        // match fragments and propagate transformations

        // 0th fragment is origin, skip it
        for (int ii = 1; ii < frame_fragments.size(); ii++)
        {
            // take the 1st of the previous fragment, and the 0th of the ii fragment
            // and use PnP to match them
        }


        // average landmarks


        // PnP to average landmarks


        /**
         *  e.g.
         * 
         *   0-4
         *   1-5
         *   2-7
         *   3-8
         * 
         *  - edges with essential
         *  - intraframes with PnP
         *  - match fragments 0:1 -> 1:0
         * 
         *  - lastly, PnP to averaged landmarks
         */

        for (Landmark& lm : landmarks)
        {
            if (lm.view_data.size() >= 3)
                lm.point3d = lm.sum_point / float(lm.view_data.size() - 1);
        }

        #ifdef DEBUG_VERBOSE
        t.stop("camera motion recovery");
        #endif
    }

    bool PoseGraph::compute_initial_pose(Camera* cam, const Camera* ref_cam)
    {
    	/**
		 * 	- first 2 frames special case
    	 * 		* essential matrix, same as before
    	 * 		* reject 2nd if over 50% features not over threshold
    	 *
    	 * 	TODO: more early termination reasons?
         *
         *  TODO: currently only-manual initial checking, 
         *          can essential matrix be used?
         * 
         *  TODO: add pointcloud registration for the initial cloud
    	 */
    	
    	// const auto rc_corr = correspondence_between_cameras(refcam, cam);
    	// std::pair<std::vector<uint32_t>, std::vector<uint32_t>> rc_corr =
        std::vector<std::pair<uint32_t, uint32_t>> rc_corr =
    		match_features_hgraphy(ref_cam->feature_descriptors, cam->feature_descriptors, ref_cam, cam);
            // match_features_bf_knn(ref_cam->feature_descriptors, cam->feature_descriptors);

        // DEBUG_visualize_feature_matches_custom(ref_cam, cam, rc_corr);

        /*         
        {
            std::vector<cv::KeyPoint> inliers1, inliers2;
            std::vector<uint32_t> inliers1_i, inliers2_i;
            std::vector<cv::DMatch> good_matches;

            for (int i = 0; i < rc_corr.size(); i++)
            {
                inliers1.emplace_back(ref_cam->kp_features[rc_corr[i].first]);
                inliers2.emplace_back(cam->kp_features[rc_corr[i].second]);

                good_matches.emplace_back(cv::DMatch(i, i, 0));
            }

            cv::Mat matchimg;
            cv::drawMatches(features::o3d_image_to_cv_image(ref_cam->rgb), inliers1, features::o3d_image_to_cv_image(cam->rgb), inliers2, good_matches, matchimg);
            cv::imshow("img", matchimg);
            cv::waitKey(0);
        }
        */

    	if (rc_corr.size() < MIN_FEATURE_COUNT)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "not enough matches: " << rc_corr.size() << ", aborting!\n";
            #endif

            remove_last_camera(cam);
    		return false;
        }

    	relative_pose_for_camera(ref_cam, cam, rc_corr);

        // refine_pose_with_ICP(cam, ref_cam);
    	landmarks_from_feature_matches(rc_corr, ref_cam, cam);

        const std::vector<uint32_t> features_negative = 
            create_match_negative(rc_corr, cam->kp_features.size());

        // camera_candidates.insert(std::make_pair(cam->id, features_negative));
        candidates_from_unmatched(features_negative, cam);

    	return true;
    }

    void PoseGraph::refine_pose_with_ICP(Camera* cam, const Camera* ref_cam)
    {
        // this method doesn't work correctly!
        assert(false);

        auto cam_cloud = rgbd::create_pcloud_from_rgbd(*cam->rgb, *cam->depth, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, false, cam->intr);
        // cam_cloud->Scale(pointcloud_scale, false);
        cam_cloud->Transform(cam->T);

        auto ref_cam_cloud = rgbd::create_pcloud_from_rgbd(*ref_cam->rgb, *ref_cam->depth, DEFAULT_DEPTH_SCALE, DEFAULT_CLOUD_FAR_CLIP, false, ref_cam->intr);
        // ref_cam_cloud->Scale(pointcloud_scale, false);
        ref_cam_cloud->Transform(ref_cam->T);


        // Colored ICP refinement
        const float voxel_size = ICP_VOXEL_SIZE;
        const float radius = ICP_NORMAL_RADIUS;

        const float distance_threshold = voxel_size * 0.4f;
        const Eigen::Matrix4d initial_transform = cam->T;

        auto cam_cloud_downsampled = cam_cloud->VoxelDownSample(voxel_size);
        auto ref_cam_cloud_downsampled = ref_cam_cloud->VoxelDownSample(voxel_size);

        cam_cloud_downsampled->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(radius, 30));
        ref_cam_cloud_downsampled->EstimateNormals(o3d::geometry::KDTreeSearchParamHybrid(radius, 30));

        // o3d::visualization::DrawGeometries({cam_cloud_downsampled, ref_cam_cloud_downsampled});

        auto result = o3d::registration::RegistrationICP(
            *cam_cloud_downsampled, *ref_cam_cloud_downsampled, distance_threshold, initial_transform,
            o3d::registration::TransformationEstimationPointToPoint()); //,
            // o3d::registration::ICPConvergenceCriteria(1e-6, 1e-6, ICP_ITER_COUNT));

        if (result.fitness_ >= ICP_FITNESS_THRESHOLD)
        {
            set_camera_T_and_P(cam, result.transformation_);
            std::cout << "camera transform differences: \n" << result.transformation_ << "\n\n" << initial_transform << "\n\n" << result.fitness_ << "\n\n";
        }
        else
        {
            // cam_cloud_downsampled->Transform(result.transformation_);
            // o3d::visualization::DrawGeometries({cam_cloud_downsampled, ref_cam_cloud_downsampled});
            std::cout << "bad fitness: " << result.fitness_ << "\n";
        }
    }

    std::vector<uint32_t> PoseGraph::create_match_negative(
        const std::vector<std::pair<uint32_t, uint32_t>> matched_features,
        const uint32_t all_features_count) const
    {
        std::vector<uint32_t> feature_negative;
        feature_negative.reserve(all_features_count - matched_features.size());

        std::vector<uint32_t> matched;
        matched.reserve(matched_features.size());

        for (int i = 0; i < matched_features.size(); i++)
        {
            matched.emplace_back(matched_features[i].second);
        }
        // std::sort(matched.begin(), matched.end());

        std::vector<uint32_t> all_features;
        all_features.reserve(all_features_count);
        for (uint32_t i = 0; i < all_features_count; i++)
        {
            all_features.emplace_back(i);
        }

        // // https://stackoverflow.com/questions/15758680/get-all-vector-elements-that-dont-belong-to-another-vector
        // std::remove_copy_if(all_features.begin(), all_features.end(), std::back_inserter(feature_negative),
        //     [&matched](const uint32_t& arg)
        //     { return (std::find(matched.begin(), matched.end(), arg) != matched.end());});

        // std::cout << "all features: " << all_features_count << ", fnegative: " << feature_negative.size() << ", matched. " << matched_features.size() << "\n";

        return set_negative_intersection(matched, all_features);
    }

    void PoseGraph::candidates_from_unmatched(const std::vector<uint32_t> unmatched_features, const Camera* cam)
    {
        const uint32_t cam_id = cam->id;
        const std::vector<cv::KeyPoint> keypoints = cam->kp_features;
        const cv::Mat descriptors = cam->feature_descriptors;
        cv::Mat cam_P;
        cv::eigen2cv(cam->P, cam_P);

        for (int i = 0; i < unmatched_features.size(); i++)
        {
            Landmark lm;
            lm.add_as_candidate(
                cam_id,
                unmatched_features[i],
                keypoints[unmatched_features[i]],
                descriptors.row(unmatched_features[i]),
                cam_P
            );

            candidates.push_back(lm);
        }
    }

    std::vector<cv::Point3f> PoseGraph::triangulate_matches(
        const std::vector<std::pair<uint32_t, uint32_t>> matches,
        const std::vector<uint32_t> first_cams, const Camera* second_cam) const
    {
        std::vector<cv::Point3f> triangulated;
        triangulated.reserve(matches.size());

        const std::vector<cv::KeyPoint> second_kps = second_cam->kp_features;
        const cv::Mat second_cam_P = [&]
        {
            cv::Mat p;
            cv::eigen2cv(second_cam->P, p);
            return p;
        }();

        for (int i = 0; i < matches.size(); i++)
        {
            const std::vector<cv::Point2f> x1 { cameras[first_cams[i]]->kp_features[matches[i].first].pt };
            const std::vector<cv::Point2f> x2 { second_kps[matches[i].second].pt };

            cv::Mat first_cam_P;
            cv::eigen2cv(cameras[first_cams[i]]->P, first_cam_P);

            cv::Mat p4d;
            cv::triangulatePoints(first_cam_P, second_cam_P, x1, x2, p4d);

            triangulated.emplace_back(cv::Point3f(
                p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
                p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
                p4d.at<float>(2, 0) / p4d.at<float>(3, 0)
            ));
        }

        // debug visualization stuff
        // std::vector<Eigen::Vector3d> points;
        // for (int i = 0; i < triangulated.size(); i++)
        //     points.emplace_back(Eigen::Vector3d(triangulated[i].x, triangulated[i].y, triangulated[i].z));

        // auto pcloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(points));
        // const auto a = o3d::geometry::TriangleMesh::CreateCoordinateFrame();
        // o3d::visualization::DrawGeometries( { pcloud, a } );
        // std::exit(0);
        
        return triangulated;
    }

    void PoseGraph::landmarks_from_feature_matches(
        const std::vector<std::pair<uint32_t, uint32_t>> matches,
        const Camera* first_cam, const Camera* second_cam)
    {
        /**
         *  triangulate matches and create new landmarks 
         */


        // std::vector<cv::DMatch> good_matches;
        // std::vector<cv::KeyPoint> inliers1, inliers2;
        // for (int i = 0; i < matches.size(); i++)
        // {
        //     inliers1.emplace_back(first_cam->kp_features[matches[i].first]);
        //     inliers2.emplace_back(second_cam->kp_features[matches[i].second]);

        //     good_matches.emplace_back(cv::DMatch(i, i, 0));
        // }

        // cv::Mat matchimg;
        // cv::drawMatches(features::o3d_image_to_cv_image(first_cam->rgb), inliers1, features::o3d_image_to_cv_image(second_cam->rgb), inliers2, good_matches, matchimg);
        // cv::imshow("img", matchimg);
        // cv::waitKey(0);

        // std::vector<Eigen::Vector3d> trpoints;

        landmarks.reserve(matches.size());

        // extend the vector to be of the same size
        const std::vector<uint32_t> first_cam_extended (matches.size(), (uint32_t)first_cam->id);
        const std::vector<cv::Point3f> triangulated = triangulate_matches(matches, first_cam_extended, second_cam);

        cv::Mat fcam_P, scam_P;
        cv::eigen2cv(first_cam->P, fcam_P);
        cv::eigen2cv(second_cam->P, scam_P);

        const std::vector<uint32_t> valid_points = filter_points_behind_camera(second_cam, triangulated);

        // create new landmarks
        for (int j = 0; j < valid_points.size(); j++)
        {
            const uint32_t i = valid_points[j];

            const cv::Point3f p3d = triangulated[i];
            
            const uint32_t ref_feature_id = matches[i].first;
            const uint32_t feature_id = matches[i].second;

            const uint32_t ref_cam_id = first_cam->id;
            const uint32_t cam_id = second_cam->id;

            const cv::KeyPoint ref_keypoint = first_cam->kp_features[ref_feature_id];
            const cv::KeyPoint keypoint = second_cam->kp_features[feature_id];

            const cv::Mat ref_descriptor = first_cam->feature_descriptors.row(ref_feature_id);
            const cv::Mat descriptor = second_cam->feature_descriptors.row(feature_id);

            Landmark lm;

            lm.add_view(ref_cam_id, ref_feature_id, ref_keypoint, ref_descriptor, fcam_P);
            lm.add_view(cam_id, feature_id, keypoint, descriptor, scam_P);
            lm.set_3d_point(p3d);
            lm.triangulation_view_pair = std::make_pair(0, 1);

            landmarks.emplace_back(lm);

            // trpoints.emplace_back(Eigen::Vector3d(p3d.x, p3d.y, p3d.z));
        }

        // auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(trpoints));
        // o3d::visualization::DrawGeometries({ cloud });
    }

    std::vector<uint32_t> PoseGraph::filter_points_behind_camera(const Camera* cam, const std::vector<cv::Point3f> points3d) const
    {
        std::vector<uint32_t> valid_points;
        valid_points.reserve(points3d.size());

        // depth test
        // return (R*X)(2) + t(2);

        for (int i = 0; i < points3d.size(); i++)
        {
            const cv::Point3f pcv = points3d[i];
            const Eigen::Vector3d p = Eigen::Vector3d(pcv.x, pcv.y, pcv.z);
            const Eigen::Vector3d pp = (cam->rotation * p) + cam->position;

            // if point has the depth of < 0 or is distance outlier
            if (pp(2) <= 0 || pp(2) > DEPTH_DISTANCE_OUTLIER)
                continue;

            valid_points.push_back(i);
        }

        #ifdef DEBUG_VERBOSE
        std::cout << "valid points: " << valid_points.size() << "; original points: " << points3d.size() << "\n";
        #endif

        return valid_points;
    }

    std::tuple<Eigen::Matrix3d, Eigen::Vector3d> PoseGraph::relative_pose_for_camera(const Camera* ref_cam, Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> feature_matches)
    {
        /**
         *  NOTE: for some reason x1 and x2 are flipped, i.e. feature_matches
         *      order is inverted.
         */

        std::vector<cv::Point2f> x1, x2;
        x1.reserve(feature_matches.size());
        x2.reserve(feature_matches.size());

        for (int i = 0; i < feature_matches.size(); i++)
        {
            const cv::KeyPoint& kp0 = ref_cam->kp_features.at(feature_matches[i].first);
            const cv::KeyPoint& kp1 = cam->kp_features.at(feature_matches[i].second);

            x1.emplace_back(kp0.pt);
            x2.emplace_back(kp1.pt);
        }

        cv::Mat mask;
        cv::Mat cv_intr;
        cv::eigen2cv(cam->intr.intrinsic_matrix_, cv_intr);

        cv::Mat E = cv::findEssentialMat(x1, x2, cv_intr, cv::RANSAC, 0.999, 0.11, mask);

        cv::Mat local_R, local_t;
        cv::recoverPose(E, x1, x2, cv_intr, local_R, local_t, mask);

        Eigen::Matrix3d dR;
        Eigen::Vector3d dt;
        cv::cv2eigen(local_R, dR);
        cv::cv2eigen(local_t, dt);

        // dt *= 0.31505707;

        cam->dR = dR;
        cam->dt = dt;

        set_camera_T_and_P(cam, dR.transpose(), -dR.transpose() * dt);

        #ifdef DEBUG_VERBOSE
        std::cout << "\033[1;31m";

        std::cout << "relative dR and dt for camera: " << ref_cam->id << "->" << cam->id << "\n";
        std::cout << "dt: " << dt.transpose() << "\n";
        #endif

        Eigen::AngleAxisd ax(dR);

        #ifdef DEBUG_VERBOSE
        std::cout << "rodrigues axis: " << ax.axis().transpose() << ", angle: " << ax.angle() * (180.0/3.14159);
        #endif

        cam->rodriques_angle = ax.angle() * (180.0/3.14159);

        #ifdef DEBUG_VERBOSE
        std::cout << "\033[0m\n\n";
        #endif

        return std::make_tuple(cam->dR, cam->dt);
    }

    bool PoseGraph::compute_pose_realtime(Camera* cam, const uint32_t time_id)
    {
        Timer t;

        std::map<std::pair<uint32_t, uint32_t>, cv::Mat> camera_homography_pairs;

        // match features against landmarks, <landmark_id, feature_id>
        std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view =
            find_landmarks_in_view(cam, camera_homography_pairs);

        #ifdef DEBUG_VERBOSE
        t.stop("find landmarks in view");
        #endif

        #ifdef DEBUG_VISUALIZE
        DEBUG_visualize_landmark_matches_against_last(cam, landmarks_in_view, 1);
        #endif
        // t.stop("draw matches visualization");

        #ifdef DEBUG_VERBOSE
        std::cout << "landmarks in view: " << landmarks_in_view.size() << "\n";
        std::cout << "landmarks: " << landmarks.size() << "\n";
        std::cout << "candidates: " << candidates.size() << "\n";
        #endif

        // create 3d-2d correspondences
        std::vector<cv::Point3f> landmark_points;
        std::vector<cv::Point2f> view_features;
        landmark_points.reserve(landmarks_in_view.size());
        view_features.reserve(landmarks_in_view.size());

        // std::vector<Eigen::Vector3d> lmdebug;

        if (landmarks_in_view.size() < reg_params.landmark_min_count)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "\033[1;31m";
            std::cout << "not enough landmarks in view: " << landmarks_in_view.size() << ", returning false\n";
            std::cout << "\033[0m\n";
            #endif

            return false;
        }

        for (int i = 0; i < landmarks_in_view.size(); i++)
        {
            landmark_points.emplace_back(landmarks[landmarks_in_view[i].first].point3d);
            view_features.emplace_back(cam->kp_features[landmarks_in_view[i].second].pt);

            // lmdebug.push_back(Eigen::Vector3d(landmark_points.back().x, landmark_points.back().y, landmark_points.back().z));
        }

        // const auto debug = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(lmdebug));
        // o3d::visualization::DrawGeometries({ debug });

        const cv::Mat cv_intr = [&]{
            cv::Mat intr;
            cv::eigen2cv(cam->intr.intrinsic_matrix_, intr);
            return intr;
        }();

        // DEBUG_visualize_landmarks(landmark_points);

        cv::Mat rcv, tcv, rcv_mat;
        std::vector<int> inliers;
        try
        {
            // use last frame as a guess
            const Camera* refcam = cameras.at(cam->id - 1);
            const auto [t_pred, r_pred, confidence] = pose_predictor.predict_next_pose(time_id);
            cv::eigen2cv(r_pred, rcv);
            cv::eigen2cv(t_pred, tcv);
            // these parameters have been figured out by painstakingly trial-and-erroring
            cv::solvePnPRansac(landmark_points, view_features, cv_intr, cam->distortion_coeffs, rcv, tcv, true, 1000, 4.0, 0.987, inliers);
        }
        catch (cv::Exception& e) {
            #ifdef DEBUG_VERBOSE
            std::cout << e.msg << "\n";
            std::cout << "lm points: " << landmark_points.size() << ", view_features: " << view_features.size() << "\n";
            for (int i = 0; i < landmark_points.size(); i++)
            {
                std::cout << landmark_points[i] << ", " << view_features[i] << "\n";
            }
            #endif

            visualize_tracks();
            std::exit(0);
        }
        
        #ifdef DEBUG_VERBOSE
        t.stop("PnPRansac");
        std::cout << "inliers size: " << inliers.size() << ", original size: " << view_features.size() << "\n";
        #endif

        if (inliers.size() < reg_params.landmark_min_count)
        {
            #ifdef DEBUG_VERBOSE
            std::cout << "\033[1;31m";
            std::cout << "adding camera failed, not enough PnP inliers for camera " << cam->id << "\n";
            std::cout << "\033[0m\n";
            #endif

            return false;
        }

        std::vector<cv::Point3f> inlier_points;
        std::vector<cv::Point2f> inlier_features;
        std::vector<std::pair<uint32_t, uint32_t>> good_lms_in_view;

        inlier_points.reserve(inliers.size());
        inlier_features.reserve(inliers.size());
        good_lms_in_view.reserve(inliers.size());

        for (int i : inliers)
        {
            inlier_points.push_back(landmark_points[i]);
            inlier_features.push_back(view_features[i]);
            good_lms_in_view.push_back(landmarks_in_view[i]);
        }

        // cv::solvePnPRefineLM(inlier_points, inlier_features, cv_intr, {}, rcv, tcv);
        // t.stop("Levenberg-Marquardt refinement");

        // extract and set the camera R and t
        {
            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            cv::Rodrigues(rcv, rcv_mat);
            cv::cv2eigen(rcv_mat, R);
            cv::cv2eigen(tcv, t);

            set_camera_T_and_P(cam, R.transpose(), -R.transpose() * t);

            #ifdef DEBUG_VERBOSE
            std::cout << "\033[1;32m";
            std::cout << "absolute world position for camera PnP: " << cam->id << "\n";
            std::cout << "t: " << cam->position.transpose() << ", t norm: " << cam->position.norm() << "\n";
            #endif

            Eigen::AngleAxisd ax(cam->rotation);

            #ifdef DEBUG_VERBOSE
            std::cout << "rodrigues axis: " << ax.axis().transpose() << ", angle: " << ax.angle() * (180.0/3.14159) << "\n";
            std::cout << "\033[0m\n";
            #endif

            cam->rodriques_angle = ax.angle() * (180.0/3.14159);
        }

        const auto [pose_score, pose_confidence]  = pose_predictor.verify_pose(time_id, cam->position, cam->rotation);

        /**
         *  low score & high confidence --> discard
         *  high score & high confidence --> keep
         * 
         *  low score & low confidence --> abort
         *  high score & low confidence --> abort
         * 
         *  reject also if pose difference is too great, compared to 1st camera position
         *      --> rejects occasional outliers due to projection error
         */
        if ((pose_score < MIN_POSE_SCORE && pose_confidence >= POSE_CONFIDENCE_THRESHOLD) ||
            (pose_confidence < POSE_CONFIDENCE_THRESHOLD) ||
            (cam->position - cameras.at(cam->id - 1)->position).norm() > cameras.at(1)->position.norm() * CAMERA_POSITION_OUTLIER_MULTIPLIER)
        {
            #ifdef DEBUG_MINIMAL
            std::cout << "\033[1;31m";
            std::cout << "adding camera failed. pose score: " << pose_score << ", pose confidence: " << pose_confidence << "\n";
            std::cout << "\033[0m\n";
            #endif

            return false;
        }

        pose_predictor.add_good_pose(time_id, cam->position, cam->rotation);

        // append landmark-feature matches to landmarks
        add_track_to_landmarks(cam, landmarks_in_view, landmark_points, inliers);

        #ifdef DEBUG_VERBOSE
        t.stop("add track to landmarks");
        #endif

        // retriangulate_landmarks(good_lms_in_view);
        // t.stop("retriangulate landmarks");

        /**
         *  NOTE:
         *      track current candidates
         *          - find unmatched non-lm features
         *          - match against candidates
         *              - add matched to candidates
         *          - remove old candidates (no track for N frames), if cannot be triangulated
         *          - triangulate succesful candidates and add to lms
         *          - add all the rest unmatched features to candidates
         */

        const std::vector<uint32_t> unmatched =
            create_match_negative(landmarks_in_view, cam->kp_features.size());

        #ifdef DEBUG_VERBOSE
        t.stop("create match negative");
        #endif

        /**
         *  TODO:
         *      optimize:
         *      - match_features_against_candidates                 (234 ms)
         *          - candidate match_features_bf since last took   (180 ms)
         *      - handle_feature_candidate_matches                  (264 ms)
         *          - to_landmark
         */

        // std::vector<uint32_t>* unmatched_features = new std::vector<uint32_t>();
        // <candidate id, current cam feature id>
        const std::vector<std::pair<uint32_t, uint32_t>> feature_candidates =
            match_features_against_candidates(cam, unmatched, camera_homography_pairs);

        #ifdef DEBUG_VERBOSE
        t.stop("match features against candidates");
        #endif

        /**
         *  TODO:
         *      track candidates until no match
         *          --> create landmark if enough views
         *          --> delete if invalid
         */

        handle_feature_candidate_matches(feature_candidates, cam, false);
        remove_old_candidates(cam->id);

        #ifdef DEBUG_VERBOSE
        t.stop("handling candidates");
        #endif
        
        // collect the subset features not yet used anywhere
        std::vector<uint32_t> subset;
        {
            subset.reserve(feature_candidates.size());
            for (const auto& elem : feature_candidates)
                subset.push_back(elem.second);
        }
        
        const std::vector<uint32_t> new_candidate_features = set_negative_intersection(subset, unmatched);
        candidates_from_unmatched(new_candidate_features, cam);

        #ifdef DEBUG_VERBOSE
        t.stop("candidate intersection and new");
        #endif

        return true;
    }

    void PoseGraph::DEBUG_visualize_landmarks(const std::vector<cv::Point3f> points) const
    {
        if (points.size() < 3)
            return;

        std::vector<Eigen::Vector3d> epoints;
        epoints.reserve(points.size());
        for (const auto p : points)
        {
            epoints.emplace_back(Eigen::Vector3d(p.x, p.y, p.z));
            std::cout << "point: " << epoints.back().transpose() << "\n";
        }

        const auto debug_cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(epoints));
        o3d::visualization::DrawGeometries({ debug_cloud });
    }

    void PoseGraph::DEBUG_visualize_landmarks(const std::vector<uint32_t> lms) const
    {
        if (lms.size() == 0)
            return;

        std::vector<cv::Point3f> points;
        points.reserve(lms.size());

        for (uint32_t i : lms)
        {
            const Landmark& lm = landmarks[i];
            points.emplace_back(lm.point3d);
        }

        DEBUG_visualize_landmarks(points);
    }

    std::vector<uint32_t> PoseGraph::set_negative_intersection(const std::vector<uint32_t> subset, const std::vector<uint32_t> all) const
    {
        if (subset.size() > all.size())
            throw std::runtime_error("subset was larger than all, aborting");
        
        std::vector<uint32_t> ret_set = all;

        for (const uint32_t s : subset)
        {
            auto iter = std::find(ret_set.begin(), ret_set.end(), s);
            if (iter != ret_set.end())
                ret_set.erase(iter);
        }

        return ret_set;
    }

    void PoseGraph::all_candidates_to_landmarks()
    {
        for (Landmark lm : candidates)
        {
            lm.to_landmark(cameras);
            lm.retriangulate_full(cameras, 0.0);
            landmarks.emplace_back(lm);
        }
    }

    void PoseGraph::retriangulate_landmarks(const std::vector<std::pair<uint32_t, uint32_t>> landmark_features)
    {
        const double lm_avg_magnitude = landmark_average_magnitude();
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < landmark_features.size(); i++)
        {
            Landmark& lm = landmarks[landmark_features[i].first];
            lm.retriangulate_full(cameras, lm_avg_magnitude);
        }

/* 
        for (int i = 0; i < landmark_features.size(); i++)
        {
            Landmark& lm = landmarks[landmark_features[i].first];
            cv::Point3f sum_point(0.f, 0.f, 0.f);
            int count = 0;

            for (int j = 0; j < lm.view_data.size(); j++)
            {
                const Camera* cam = cameras.at(lm.view_data[j].camera_id);

                for (int k = j; k < lm.view_data.size(); k++)
                {
                    const cv::Point3f p3d = lm.get_triangulated(j, k);

                    // point is out of bounds
                    // if (sqrt(p3d.dot(p3d)) > TRIANGULATED_POINT_OUTLIER_NORM)
                    // const double point_depth = math::depth_from_RtX(cam->rotation, cam->position, Eigen::Vector3d(p3d.x, p3d.y, p3d.z));
                    // if (point_depth <= 0 || point_depth > DEPTH_DISTANCE_OUTLIER)
                    //     continue;

                    if (!lm.to_landmark(cameras, lm_avg_magnitude))
                    {
                        continue;
                    }

                    sum_point += p3d;
                    count++;
                }
            }

            if (count > 0)
            {
                // std::cout << "original: " << lm.point3d << "; new point: " << sum_point / (float)count << "\n";
                lm.set_3d_point(sum_point / (float)count);
            }
        } */
    }

    void PoseGraph::retriangulate_landmarks()
    {
        const double lm_avg_3d = landmark_average_magnitude();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < landmarks.size(); i++)
        {
            landmarks[i].retriangulate_full(cameras, lm_avg_3d);
        }
    }

    void PoseGraph::remove_old_candidates(const uint32_t cam_id)
    {
        std::vector<uint32_t> to_remove_candidates;
        std::vector<uint32_t> debug_landmarks;

        const double lm_avg_dist = landmark_average_magnitude();

        for (int i = 0; i < candidates.size(); i++)
        {
            Landmark& cd = candidates[i];

            // candidate lost tracking and didn't reach required length --> delete
            if (abs((int)cd.view_data.back().camera_id - (int)cam_id) >= INVALID_CANDIDATE_CAMERA_TEMPORAL_DIFF &&
                cd.view_data.size() < reg_params.candidate_chain_len)
            {
                to_remove_candidates.push_back(i);
            }
            // candidate lost tracking but reached the required length --> to landmark
            // or enough movement between first and last
            else if ((abs((int)cd.view_data.back().camera_id - (int)cam_id) >= INVALID_CANDIDATE_CAMERA_TEMPORAL_DIFF &&
                cd.view_data.size() >= reg_params.candidate_chain_len) ||
                candidate_movement_large_enough(cd))
            {
                cd.to_landmark(cameras);
                cd.retriangulate_full(cameras, lm_avg_dist);
                landmarks.push_back(cd);
                debug_landmarks.push_back(landmarks.size() - 1);
                
                // const cv::Point2f front_pt = cd.view_data.front().feature_kp.pt;
                // const cv::Point2f back_pt = cd.view_data.back().feature_kp.pt;

                // // triangulate and create a landmark
                // if (cd.to_landmark(cameras, lm_avg_dist))
                // {
                //     cd.retriangulate_full(cameras, lm_avg_dist);
                //     landmarks.push_back(cd);
                //     debug_landmarks.push_back(landmarks.size() - 1);
                // }

                to_remove_candidates.push_back(i);
            }
        }

        // DEBUG_visualize_majority_candidate_matches(debug_landmarks, 100, false);
        // DEBUG_visualize_landmarks(debug_landmarks);
        remove_candidates(to_remove_candidates);
    }

    void PoseGraph::handle_feature_candidate_matches(const std::vector<std::pair<uint32_t, uint32_t>> matches, const Camera* cam, const bool check_for_lm)
    {
        cv::Mat cam_P;
        cv::eigen2cv(cam->P, cam_P);
        const uint32_t cam_id = cam->id;

        std::vector<uint32_t> to_remove_candidates;
        if (check_for_lm)
            to_remove_candidates.reserve(matches.size());

        std::vector<uint32_t> debug_landmarks;

        const double lm_avg_dist = landmark_average_magnitude();

        for (int i = 0; i < matches.size(); i++)
        {
            const uint32_t candidate_id = matches[i].first;
            const uint32_t feature_id = matches[i].second;

            Landmark& cd = candidates[candidate_id];
            cd.add_view(
                cam_id,
                feature_id,
                cam->kp_features[feature_id],
                cam->feature_descriptors.row(feature_id),
                cam_P,
                cameras,
                lm_avg_dist,
                false
            );


            if (cd.view_data.size() >= reg_params.candidate_chain_len && check_for_lm)
            {
                cd.to_landmark(cameras);
                cd.retriangulate_full(cameras, lm_avg_dist);
                landmarks.push_back(cd);
                debug_landmarks.push_back(landmarks.size() - 1);

                // const cv::Point2f front_pt = cd.view_data.front().feature_kp.pt;
                // const cv::Point2f back_pt = cd.view_data.back().feature_kp.pt;

                // // const float feature_diff_magnitude = sqrt((back_pt - front_pt).dot(back_pt - front_pt));

                // // triangulate and create a landmark
                // if (cd.to_landmark(cameras))
                // {
                //     landmarks.push_back(cd);
                //     debug_landmarks.push_back(landmarks.size() - 1);
                // }

                // either way, the candidate will be removed
                to_remove_candidates.push_back(matches[i].first);
            }
        }

        // DEBUG_visualize_majority_candidate_matches(debug_landmarks);
        // DEBUG_visualize_landmarks(debug_landmarks);

        if (check_for_lm)
            remove_candidates(to_remove_candidates);
    }

    std::vector<std::pair<uint32_t, uint32_t>> PoseGraph::filter_candidate_matches_homography(
        const std::vector<std::pair<uint32_t, uint32_t>> unfiltered,
        const Camera* cam)
    {
        // group the candidate by camera (min n (4) lms per view?), compute homography matrix
        // by-view, filter outliers by-view

        // <camera_id, <corresponding candidate_match_ids>>
        // the indexes in .second are indexes to unfiltered
        std::map<uint32_t, std::vector<uint32_t>> cam_grouped_matches;

        // if any candidate's views camera in cam_grouped_matches
        // --> add, else create from 0th
        for (uint32_t i = 0; i < unfiltered.size(); i++)
        {
            const Landmark& lm = candidates[unfiltered[i].first];

            std::map<uint32_t, std::vector<uint32_t>>::iterator cam_iter =
                cam_grouped_matches.find(lm.view_data.front().camera_id);
            
            // found, append the candidate_id 
            if (cam_iter != cam_grouped_matches.end())
            {
                cam_iter->second.push_back(i);
            }
            // not found, create new
            else
            {
                cam_grouped_matches.insert(
                    std::pair<uint32_t, std::vector<uint32_t>>(
                        lm.view_data.front().camera_id,
                        { i }
                    )
                );   
            }
        }

        std::vector<std::pair<uint32_t, uint32_t>> filtered_matches;
        filtered_matches.reserve(unfiltered.size());

        // filter with homography
        for (const auto it : cam_grouped_matches)
        {
            if (it.second.size() < INVALID_CANDIDATE_CAMERA_TEMPORAL_DIFF)
                continue;

            std::vector<cv::Point2f> fpoints1, fpoints2;
            fpoints1.reserve(it.second.size());
            fpoints2.reserve(it.second.size());

            std::vector<cv::DMatch> good_matches;
            std::vector<cv::KeyPoint> fkp1, fkp2;

            // collect fpoints1 and fpoints2
            for (uint32_t i : it.second)
            {
                fpoints1.push_back(candidates[unfiltered[i].first].view_data.front().feature_kp.pt);
                fpoints2.push_back(cam->kp_features[unfiltered[i].second].pt);

                fkp1.push_back(candidates[unfiltered[i].first].view_data.front().feature_kp);
                fkp2.push_back(cam->kp_features[unfiltered[i].second]);
            }

            cv::Mat inlier_mask, homography;
            std::vector<cv::DMatch> inlier_matches;

            homography = findHomography(fpoints1, fpoints2, cv::RANSAC, ransac_thresh, inlier_mask);

            int l_i = 0;
            for (size_t i = 0; i < fpoints1.size(); i++)
            {
                cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
                col.at<double>(0) = fpoints1[i].x;
                col.at<double>(1) = fpoints2[i].y;
                col = homography * col;
                col /= col.at<double>(2);

                const double dist = sqrt(pow(col.at<double>(0) - fpoints2[i].x, 2) + pow(col.at<double>(1) - fpoints2[i].y, 2));
                if (dist < CANDIDATE_INLIER_THRESHOLD)
                {
                    filtered_matches.push_back(unfiltered[it.second[i]]);

                    good_matches.emplace_back(cv::DMatch(i, i, 0));
                    l_i++;
                }
            }

            // debug visualize matches
            {
                if (good_matches.size() == 0)
                    continue;

                auto ref_rgb = cameras[it.first]->rgb;

                cv::Mat matchimg;
                cv::drawMatches(rgbd::o3d_image_to_cv_image(ref_rgb), fkp1, rgbd::o3d_image_to_cv_image(cam->rgb), fkp2, good_matches, matchimg);
                cv::imshow("img", matchimg);
                cv::waitKey(0);
            }
        }

        filtered_matches.shrink_to_fit();

        #ifdef DEBUG_VERBOSE
        std::cout << "sizes : " << filtered_matches.size() << ", " << unfiltered.size() << "\n";
        #endif

        return filtered_matches;
    }

    std::vector<uint32_t> PoseGraph::find_negative_features_from_unmatched(
            const std::vector<std::pair<uint32_t, uint32_t>> feature_candidate_matches,
            const std::vector<uint32_t> landmark_unmatched_features) const
    {
        /**
         *  finds the features not in landmark_unmatched_features but in 
         *  feature_candidate_matches
         * 
         *  feature_candidate_matches should always be smaller
         */

        if (landmark_unmatched_features.size() < feature_candidate_matches.size())
        {
            std::cout << "lm unmatched features size: " << landmark_unmatched_features.size() << ", fcm size: " << feature_candidate_matches.size() << "\n"; 
            throw std::runtime_error("landmark_unmatched_features was larger than feature_candidate_matches, impossible case! aborting");
        }

        std::vector<uint32_t> negative;
        negative.reserve(landmark_unmatched_features.size() - feature_candidate_matches.size());

        std::vector<uint32_t> feature_candidate_matches_collected;
        feature_candidate_matches_collected.reserve(feature_candidate_matches.size());
        for (int i = 0; i < feature_candidate_matches.size(); i++)
            feature_candidate_matches_collected.emplace_back(feature_candidate_matches[i].second);

        // https://stackoverflow.com/questions/15758680/get-all-vector-elements-that-dont-belong-to-another-vector
        std::remove_copy_if(landmark_unmatched_features.begin(), landmark_unmatched_features.end(), std::back_inserter(negative),
            [&feature_candidate_matches_collected](const uint32_t& arg)
            { return (std::find(feature_candidate_matches_collected.begin(), feature_candidate_matches_collected.end(), arg) != feature_candidate_matches_collected.end());});

        return negative;
    }

    void PoseGraph::remove_candidates(std::vector<uint32_t> matched_candidates)
    {
        // sort and reverse the candidates for easier removing
        std::sort(matched_candidates.begin(), matched_candidates.end());
        std::reverse(matched_candidates.begin(), matched_candidates.end());

        for (int i : matched_candidates)
            candidates.erase(candidates.begin() + i);
    }

    std::vector<uint32_t> PoseGraph::add_landmarks_from_candidates(const Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> candidate_matches)
    {
        /**
         *  if possible, triangulate and add to landmarks
         *  else add view to landmark
         */

        std::vector<uint32_t> succesfull_candidates;
        succesfull_candidates.reserve(candidate_matches.size());

        const cv::Mat cam_P = [&]{
            cv::Mat p;
            cv::eigen2cv(cam->P, p);
            return p;
        }();
        const uint32_t cam_id = cam->id;

        std::vector<Eigen::Vector3d> debug_points;
        std::vector<cv::Point2f> debug_features;

        const double lm_avg_magnitude = landmark_average_magnitude();

        for (int i = 0; i < candidate_matches.size(); i++)
        {
            const uint32_t candidate_id = candidate_matches[i].first;
            Landmark& candidate = candidates[candidate_id];
            const uint32_t feature_id = candidate_matches[i].second;

            const std::vector<cv::Point2f> x1 = { candidate.view_data[0].feature_kp.pt };
            const std::vector<cv::Point2f> x2 = { cam->kp_features[feature_id].pt };

            // skip triangulation and do not add if deviation not big enough
            if (abs(x1[0].x - x2[0].x) < FEATURE_TRIANGULATION_MIN_DIFF && 
                abs(x1[0].y - x2[0].y) < FEATURE_TRIANGULATION_MIN_DIFF)
            {
                candidate.add_view(
                    cam_id,
                    feature_id,
                    cam->kp_features[feature_id],
                    cam->feature_descriptors.row(feature_id),
                    cam_P,
                    cameras,
                    lm_avg_magnitude,
                    true
                );
                continue;
            }
            succesfull_candidates.emplace_back(candidate_id);

            const cv::Mat firstcam_P = candidate.view_data[0].camera_P;
            
            // std::cout << "id: " << candidate.view_data[0].camera_id << "\n" << firstcam_P << "\n\n" << cam_P << "\n\n\n";
            
            cv::Mat p4d;
            cv::triangulatePoints(firstcam_P, cam_P, x1, x2, p4d);

            const cv::Point3f p3d(
                p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
                p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
                p4d.at<float>(2, 0) / p4d.at<float>(3, 0)
            );

            // discard the candidate/landmark if triangulation failed
            if (sqrt(p3d.dot(p3d)) > TRIANGULATED_POINT_OUTLIER_NORM)
                continue;

            Landmark lm = candidate;

            lm.add_view(
                cam_id,
                feature_id,
                cam->kp_features[feature_id],
                cam->feature_descriptors.row(feature_id),
                cam_P,
                cameras,
                lm_avg_magnitude,
                true
            );

            lm.set_3d_point(p3d);
            landmarks.emplace_back(lm);

            debug_points.emplace_back(Eigen::Vector3d(p3d.x, p3d.y, p3d.z));
            debug_features.emplace_back(x2[0]);
        }

        /* if (debug_points.size() > 10)
        {
            auto debug_cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(debug_points));
            std::shared_ptr<o3d::geometry::TriangleMesh> camera_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            o3d::io::ReadTriangleMeshFromOBJ(reg_params.assets_path + DEBUG_CAMERA_PATH, *camera_mesh, false);
            camera_mesh->Transform(cam->T);
            DEBUG_visualize_features(cam->rgb, debug_features);
            o3d::visualization::DrawGeometries({debug_cloud, camera_mesh});
        } */

        succesfull_candidates.shrink_to_fit();
        return succesfull_candidates;
    }

    std::vector<std::pair<uint32_t, uint32_t>> PoseGraph::match_features_against_candidates(const Camera* cam,
        const std::vector<uint32_t> matches_negative,
        std::map<std::pair<uint32_t, uint32_t>, cv::Mat>& camera_hgraphy_pairs) const
    {
        if (candidates.size() == 0 || matches_negative.size() == 0)
            return std::vector<std::pair<uint32_t, uint32_t>>();

        // create candidate descriptors
        const cv::Mat candidate_descriptors = create_candidate_descriptor();

        // check for empty
        if (candidate_descriptors.rows == 0)
            return std::vector<std::pair<uint32_t, uint32_t>>();

        // collect unmatched feature descriptors
        cv::Mat unmatched_descriptors (matches_negative.size(), cam->feature_descriptors.cols, CV_32F);
        const cv::Mat cam_kp_descriptors = cam->feature_descriptors;

        for (int i = 0; i < matches_negative.size(); i++)
        {
            const float* const vaptr = cam_kp_descriptors.ptr<float>(matches_negative[i], 0);
            
            float* fptr = unmatched_descriptors.ptr<float>(i, 0);
            for (int j = 0; j < unmatched_descriptors.cols; j++)
            {
                fptr[j] = vaptr[j];
            }
        }

        Timer t;
        // <candidate_id, index in matches_negative>
        std::vector<std::pair<uint32_t, uint32_t>> candidate_matches = 
            match_features_bf(candidate_descriptors, unmatched_descriptors);
        
        #ifdef DEBUG_VERBOSE
        t.stop("candidate match_features_bf");
        #endif

        // unravel candidate_matches to be 
        // <candidate_id, feature_id>
        for (int i = 0; i < candidate_matches.size(); i++)
        {
            const uint32_t second_id = candidate_matches[i].second;
            candidate_matches[i].second = matches_negative[second_id];
        }

        // homography filtering. Marginally slower, but yields consistently more robust results
        const auto good_matches = cdlm_pairwise_homography_filter(cam, candidate_matches, candidates, camera_hgraphy_pairs);

        #ifdef DEBUG_VERBOSE
        t.stop("candidate hgraphy filtering");
        #endif
        // DEBUG_visualize_candidate_matches(cam, good_matches, 1);

        return good_matches;
    }

    cv::Mat PoseGraph::create_candidate_descriptor() const
    {
        cv::Mat descriptor (candidates.size(), candidates[0].view_data[0].feature_descriptor.cols, CV_32F);

        // set the descriptor rows to be landmark descriptors
        for (int i = 0; i < candidates.size(); i++)
        {
            const cv::Mat view_descriptor = candidates[i].view_data.back().feature_descriptor;
            const float* vaptr = view_descriptor.ptr<float>(0, 0);
            
            float* fptr = descriptor.ptr<float>(i, 0);
            for (int j = 0; j < descriptor.cols; j++)
            {
                fptr[j] = vaptr[j];
            }
        }

        return descriptor;
    }

    void PoseGraph::add_track_to_landmarks(const Camera* cam,
        const std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view,
        const std::vector<cv::Point3f> points3d,
        const std::vector<int> inliers)
    {
        const uint32_t cam_id = cam->id;
        cv::Mat cam_P;
        cv::eigen2cv(cam->P, cam_P);

        const double lm_avg_magnitude = landmark_average_magnitude();

        for (int i : inliers)
        {
            const uint32_t landmark_id = landmarks_in_view[i].first;
            const uint32_t feature_id = landmarks_in_view[i].second;
            // const cv::Point3f p3d = points3d[i];
            const cv::KeyPoint kpf = cam->kp_features[feature_id];
            const cv::Mat descriptor = cam->feature_descriptors.row(feature_id);

            landmarks.at(landmark_id).add_view(
                cam_id,
                feature_id,
                kpf,
                descriptor,
                cam_P,
                cameras,
                lm_avg_magnitude,
                true
            );
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> 
        PoseGraph::find_landmarks_in_view(const Camera* cam,
        std::map<std::pair<uint32_t, uint32_t>, cv::Mat>& camera_hgraphy_pairs,
        const uint32_t view_offset) const
    {
        Timer t;
        
        const cv::Mat landmark_descriptor = create_landmark_descriptor(view_offset);

        #ifdef DEBUG_VERBOSE
        t.stop("create landmark descriptor");
        #endif

        // < lm_id, f_id >
        const std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view = 
            match_features_bf(landmark_descriptor, cam->feature_descriptors);

        #ifdef DEBUG_VERBOSE
        t.stop("match landmarks against features");
        #endif

        const auto good_landmarks_in_view = cdlm_pairwise_homography_filter(cam, landmarks_in_view, landmarks, camera_hgraphy_pairs, true);

        #ifdef DEBUG_VERBOSE
        t.stop("landmark homography filtering");
        #endif

        return good_landmarks_in_view;
    }

    cv::Mat PoseGraph::create_landmark_descriptor(const uint32_t view_offset) const
    {
        if (landmarks.size() == 0)
        {
            throw std::runtime_error("cannot create a descriptor because there are no landmarks!");
        }

        cv::Mat descriptor (landmarks.size(), landmarks[0].view_data[0].feature_descriptor.cols, CV_32F);

        // set the descriptor rows to be landmark descriptors
        for (int i = 0; i < landmarks.size(); i++)
        {

            const cv::Mat view_average_desc = landmarks[i].view_data[landmarks[i].view_data.size() - 1 - 
                (landmarks[i].view_data.size() > view_offset ? view_offset : 0)
            ].feature_descriptor;
                // compute_view_average_descriptor();
            const float* vaptr = view_average_desc.ptr<float>(0, 0);
            
            float* fptr = descriptor.ptr<float>(i, 0);
            for (int j = 0; j < descriptor.cols; j++)
            {
                fptr[j] = vaptr[j];
            }
        }

        return descriptor;
    }

    std::shared_ptr<o3d::geometry::PointCloud> PoseGraph::landmarks_to_pointcloud() const
    {
        std::vector<Eigen::Vector3d> points3d, colors;
        points3d.reserve(landmarks.size());
        colors.reserve(landmarks.size());

        for (int i = 0; i < landmarks.size(); i++)
        {
            const cv::Point3f p3d = landmarks[i].point3d;
            const Eigen::Vector3d p3de (p3d.x, p3d.y, p3d.z);

            // if (p3de.norm() > 30.0)
            //     continue;
            
            points3d.emplace_back(p3de);
            colors.emplace_back(landmarks[i].point_color);
        }

        auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(points3d));
        cloud->colors_ = colors;
        return cloud;
    }

    void PoseGraph::create_pointclouds_from_depths(const bool only_scale)
    {
        if (cameras.empty())
            throw std::runtime_error("cannot create pointclouds from non-existent cameras!");
        
        if (!only_scale)
            depth_pointclouds.clear();

        // calculate scale for depthclouds
    	// gather all landmarks which have .front().camera_index  == 0
        std::vector<cv::Point3f> zero_view_points;
        std::vector<cv::Point2f> zero_view_features;
        for (int i = 0; i < landmarks.size(); i++)
        {
            const Landmark& lm = landmarks[i];
            if (lm.view_data.front().camera_id == 0)
            {
                zero_view_features.push_back(lm.view_data.front().feature_kp.pt);
                zero_view_points.push_back(lm.point3d);
            }
        }

        uint32_t count = 0;
        double cloud_scale = 0.0;

        const Camera* cam = cameras.at(0);

        for (int i = 0; i < zero_view_points.size(); i++)
        {
            const double depth = (double)(*cam->depth->PointerAt<uint16_t>(
                (int)zero_view_features[i].x, (int)zero_view_features[i].y)) / 1000.0;
            
            // do not account for occluded pixels
            if (depth < DEPTH_NEAR_CLIPPING)
                continue;

            const Eigen::Vector3d depth_point = rgbd::triangulate_point(
                Eigen::Vector2d(zero_view_features[i].x, zero_view_features[i].y),
                depth, cam->intr);
            
            const Eigen::Vector4d depth_point_transformed = cam->T * Eigen::Vector4d(depth_point.x(), depth_point.y(), depth_point.z(), 0.0);
            const Eigen::Vector3d dpoint (depth_point_transformed.x(), depth_point_transformed.y(), depth_point_transformed.z());

            cloud_scale += 
                Eigen::Vector3d(zero_view_points[i].x, zero_view_points[i].y, zero_view_points[i].z).norm() / 
                dpoint.norm();

            count++;
        }

        this->pointcloud_scale = cloud_scale / (double)count;
        #ifdef DEBUG_VERBOSE
        std::cout << "pointcloud scale: " << pointcloud_scale << "\n";
        #endif

        if (only_scale)
            return;

        // project depth images to pointclouds
        for (int i = 0; i < cameras.size(); i++)
        {
            if (cameras[i]->depth == nullptr || cameras[i]->position.hasNaN() || cameras[i]->position.norm() > 1000.0)
                continue;

            auto cloud = rgbd::create_pcloud_from_rgbd(*cameras[i]->rgb, *cameras[i]->depth, DEFAULT_DEPTH_SCALE, reg_params.depth_far_clip, false, cameras[i]->intr);

            if (cloud->points_.size() == 0)
            {
                std::cout << "skip camera " << i << " due to no points\n";
                continue;
            }

            cloud->Scale(this->pointcloud_scale, false);
            cloud->Transform(cameras[i]->T);

            depth_pointclouds.emplace_back(cloud);
        }
    }

    void PoseGraph::remove_last_camera(Camera* cam)
    {
        const uint32_t cam_id = cam->id;
        if (cameras.back()->id == cam_id)
        {
            cameras.pop_back();
        }
    }

    std::vector<std::shared_ptr<Camera>> PoseGraph::get_cameras_copy() const
    {
        std::vector<std::shared_ptr<Camera>> rcameras;

        for (const auto c : cameras)
        {
            rcameras.emplace_back(std::make_shared<Camera>(Camera(*c)));
        }

        return rcameras;
    }

    void PoseGraph::bundle_adjust()
    {
        /**
         *  MAYBE:
         *      http://docs.ros.org/en/melodic/api/gtsam/html/SFMExample__SmartFactor_8cpp_source.html
         *      https://github.com/nghiaho12/SFM_example/blob/3b95176d9758752cbb1ebb7fa0050ded0b85e950/src/main.cpp#L352
         * 
         */
        
        #ifdef DEBUG_MINIMAL
        std::cout << "begin bundle adjustment \n";
        #endif

        gtsam::Values result;

        // add camera calibration matrix

        const double cx = cameras.front()->intr.intrinsic_matrix_.coeff(0, 2);
        const double cy = cameras.front()->intr.intrinsic_matrix_.coeff(1, 2);
        const double fx = cameras.front()->intr.intrinsic_matrix_.coeff(0, 0);
        const double fy = cameras.front()->intr.intrinsic_matrix_.coeff(1, 1);

        gtsam::Cal3_S2 K(fx ,fy, 0, cx, cy);
        const gtsam::noiseModel::Isotropic::shared_ptr measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, ADJUST_2DPOINT_NOISE);

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;

        // add camera indexes and transformations (R, t)
        for (int i = 0; i < cameras.size(); i++)
        {
            const Camera* cam = cameras.at(i);

            const gtsam::Rot3 R (
                cam->T.coeff(0, 0), cam->T.coeff(0, 1), cam->T.coeff(0, 2),
                cam->T.coeff(1, 0), cam->T.coeff(1, 1), cam->T.coeff(1, 2),
                cam->T.coeff(2, 0), cam->T.coeff(2, 1), cam->T.coeff(2, 2)
            );

            const gtsam::Point3 t (
                cam->T.coeff(0, 3),
                cam->T.coeff(1, 3),
                cam->T.coeff(2, 3)
            );

            const gtsam::Pose3 pose (R, t);

            if (i == 0)
            {
                gtsam::noiseModel::Diagonal::shared_ptr pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << gtsam::Vector3::Constant(ADJUST_3DPOINT_NOISE), gtsam::Vector3::Constant(ADJUST_3DPOINT_NOISE)).finished());

                graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', 0), pose, pose_noise);
            }

            initial.insert(gtsam::Symbol('x', cam->id), pose);
        }

        // add landmarks
        for (int i = 0; i < landmarks.size(); i++)
        {
            const Landmark& lm = landmarks[i];

            for (int j = 0; j < lm.view_data.size(); j++)
            {
                const cv::Point2f& ptcv = lm.view_data[j].feature_kp.pt;
                const gtsam::Point2 pt (ptcv.x, ptcv.y);

                graph.emplace_shared<gtsam::GeneralSFMFactor2<gtsam::Cal3_S2>>(pt, measurement_noise, gtsam::Symbol('x', lm.view_data[j].camera_id),
                    gtsam::Symbol('l', i), gtsam::Symbol('K', 0));
            }
        }

        initial.insert(gtsam::Symbol('K', 0), K);

        gtsam::noiseModel::Diagonal::shared_ptr cal_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(5) << 0.0, 0.0, 0, 0.0, 0.0).finished());
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Cal3_S2>>(gtsam::Symbol('K', 0), K, cal_noise);

        for (int i = 0; i < landmarks.size(); i++)
        {
            const Landmark& lm = landmarks[i];

            const cv::Point3f p = lm.point3d;
            initial.insert<gtsam::Point3>(gtsam::Symbol('l', i), gtsam::Point3(p.x, p.y, p.z));

            if (i == 0)
            {
                // maximum noise for pose, 0.3m -> 30cm (assumed)
                // gtsam::noiseModel::Isotropic::shared_ptr point_noise = gtsam::noiseModel::Isotropic::Sigma(3, ADJUST_3DPOINT_NOISE);
                gtsam::noiseModel::Isotropic::shared_ptr point_noise = gtsam::noiseModel::Isotropic::Sigma(3, ADJUST_3DPOINT_NOISE);
                graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(gtsam::Symbol('l', i), gtsam::Point3(p.x, p.y, p.z), point_noise);
            }
        }

        #ifdef DEBUG_MINIMAL
        std::cout << "data added to bundle adjustment, adjusting...\n";
        #endif

        result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
        #ifdef DEBUG_VERBOSE
        std::cout << "\n" << "initial graph error: " << graph.error(initial) << "\n" << "final graph error: " << graph.error(result) << "\n";
        #endif

        // std::vector<Eigen::Vector3d> old_positions;

        // set new camera poses
        for (int i = 0; i < cameras.size(); i++)
        {
            const Eigen::Matrix3d R = result.at<gtsam::Pose3>(gtsam::Symbol('x', i)).rotation().matrix();
            const Eigen::Vector3d t = result.at<gtsam::Pose3>(gtsam::Symbol('x', i)).translation();

            // std::cout << "camera adjust movement: " << (cameras[i]->position - (-R.transpose() * t)).norm() << "\n";
            // old_positions.push_back(cameras[i]->position);

            set_camera_T_and_P(cameras[i], R, t);
        }

        // for (int i = 0; i < old_positions.size(); i++)
        // {
        //     std::cout << "old: " << old_positions[i].transpose() << ", new: " << cameras[i]->position.transpose() << "\n";
        // }

        // retriangulate_landmarks();
    }

    void PoseGraph::recompute_camera_poses_landmark_PnP()
    {
        /**
         *  for each cameras, collect applying 2d-3d feature-landmark 
         *  correspondences, PnP
         */

        for (int ii = 0; ii < cameras.size(); ii++)
        {
            Camera* cam = cameras.at(ii);

            std::map<std::pair<uint32_t, uint32_t>, cv::Mat> camera_homography_pairs;

            // match features against landmarks, <landmark_id, feature_id>
            std::vector<std::pair<uint32_t, uint32_t>> landmarks_in_view =
                find_landmarks_in_view(cam, camera_homography_pairs);

            // collect 3d landmarks and 2d feature points
            std::vector<cv::Point2f> view_features;
            std::vector<cv::Point3f> landmark_points;

            std::vector<Eigen::Vector3d> points3d;

            for (const auto& lfpair : landmarks_in_view)
            {
                const auto p3d = landmarks.at(lfpair.first).point3d;
                if (p3d == cv::Point3f(0.0, 0.0, 0.0))
                    continue;

                landmark_points.push_back(p3d);
                view_features.push_back(cam->kp_features[lfpair.second].pt);
                points3d.push_back(Eigen::Vector3d(p3d.x, p3d.y, p3d.z));

                // std::cout << view_features.back() << " : " << landmark_points.back() << "\n";
            }

            // DEBUG_visualize_features(cam->rgb, view_features);
            // DEBUG_visualize_landmarks(landmark_points);

            // {
            //     std::shared_ptr<o3d::geometry::TriangleMesh> camera_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            //             o3d::io::ReadTriangleMeshFromOBJ(reg_params.assets_path + DEBUG_CAMERA_PATH, *camera_mesh, false);
            //             camera_mesh->Transform(cam->T);
            
            //     auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(points3d));
            //     o3d::visualization::DrawGeometries({ cloud, camera_mesh });
            // }

            const cv::Mat cv_intr = [&]{
                cv::Mat intr;
                cv::eigen2cv(cam->intr.intrinsic_matrix_, intr);
                return intr;
            }();

            cv::Mat rcv_rodr, rcv, tcv, inliers;
            // cv::eigen2cv(cam->rotation, rcv);
            // cv::eigen2cv(cam->position, tcv);

            // these parameters have been figured out by painstakingly trial-and-erroring
            const bool retval = cv::solvePnPRansac(landmark_points, view_features, cv_intr, cam->distortion_coeffs, rcv, tcv, false, 1000, 4.0, 0.987, inliers);


            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            cv::Rodrigues(rcv, rcv_rodr);
            cv::cv2eigen(rcv_rodr, R);
            cv::cv2eigen(tcv, t);

            // std::cout << "inliers: " << inliers.size << " and retval " << retval << " t " << t.transpose() <<  "\n";

            // const Eigen::Matrix3d old_rot = cam->rotation;

            set_camera_T_and_P(cam, R.transpose(), -R.transpose() * t);

            // std::cout << cam->rotation << "\n\n" << old_rot << "\n\n" << cam->position.transpose() << "\n";

            // {
            //     std::shared_ptr<o3d::geometry::TriangleMesh> camera_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            //             o3d::io::ReadTriangleMeshFromOBJ(reg_params.assets_path + DEBUG_CAMERA_PATH, *camera_mesh, false);
            //             camera_mesh->Transform(cam->T);
            
            //     auto cloud = std::make_shared<o3d::geometry::PointCloud>(o3d::geometry::PointCloud(points3d));
            //     o3d::visualization::DrawGeometries({ cloud, camera_mesh });
            // }
        }
    }

    void PoseGraph::visualize_tracks() const
    {
        std::vector<std::shared_ptr<o3d::geometry::TriangleMesh>> camera_meshes;
        std::vector<std::shared_ptr<const o3d::geometry::Geometry>> world_meshes;
        for (int i = 0; i < cameras.size(); i++)
        {
            const Camera* c = cameras.at(i);

            if (c->position.hasNaN() || c->position.norm() > 200)
                continue;

            std::shared_ptr<o3d::geometry::TriangleMesh> camera_mesh = std::make_shared<o3d::geometry::TriangleMesh>(o3d::geometry::TriangleMesh());
            o3d::io::ReadTriangleMeshFromOBJ(reg_params.assets_path + DEBUG_CAMERA_PATH, *camera_mesh, false);
            
            if (pointcloud_scale == 0.0f)
                camera_mesh->Scale(1.0f);
            else
                camera_mesh->Scale(pointcloud_scale * CAMERA_SCALE);

            camera_mesh->Transform(c->T);

            // camera_mesh->texture_.Clear();
            // camera_mesh->PaintUniformColor(index2color(i));

            camera_meshes.push_back(camera_mesh);
            world_meshes.push_back(camera_mesh);

            #ifdef DEBUG_MINIMAL
            std::cout << "camera " << c->id << " position: " << c->position.transpose() << "\n";
            #endif
        }

        #ifdef DEBUG_VERBOSE
        std::cout << "triangulated clouds size: " << ptrclouds.size() << ", depth clouds: " << depth_pointclouds.size() << "\n";
        #endif

        /* 
        for (int i = 0; i < ptrclouds.size(); i++)
        {
            break;

            auto cloud = ptrclouds[i];
            if (cloud->points_.size() < 10)
                continue;

            std::cout << "cloud points: " << cloud->points_.size() << "\n";

            if (cloud->colors_.size() == 0)
            {
                std::cout << "featurecloud colors painted\n";
                cloud->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
            }

            for (int p3d_i = 0; p3d_i < cloud->points_.size(); p3d_i++)
            {
                if (cloud->points_.at(p3d_i).norm() > 60.0)
                    cloud->points_.at(p3d_i) = Eigen::Vector3d::Zero();
            }

            world_meshes.push_back(cloud);
        }
        */

        for (int i = 0; i < depth_pointclouds.size(); i++)
        {
            // break;

            auto temp_cloud = depth_pointclouds[i];
            world_meshes.push_back(temp_cloud);
        }

        const auto lm_cloud = landmarks_to_pointcloud();
        // world_meshes.push_back(lm_cloud);

        world_meshes.push_back(o3d::geometry::TriangleMesh::CreateCoordinateFrame());

        o3d::visualization::DrawGeometries(world_meshes, "camera track", 1920, 1080, 180, 120);
    }
}
