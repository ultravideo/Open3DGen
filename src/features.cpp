#include "features.h"


namespace stitcher3d::features
{   

    using namespace rgbd;

    std::vector<Eigen::Vector2d> undistort_features(const std::vector<cv::KeyPoint>& features, 
        const std::vector<double>& dist_coeffs, const o3d::camera::PinholeCameraIntrinsic& intr)
    {
        std::vector<Eigen::Vector2d> eigen_features;
        eigen_features.reserve(features.size());
        for (const auto& f : features)
            eigen_features.emplace_back(Eigen::Vector2d(f.pt.x, f.pt.y));

        return undistort_features(eigen_features, dist_coeffs, intr);
    }

    std::vector<Eigen::Vector2d> undistort_features(const std::vector<Eigen::Vector2d>& features, 
        const std::vector<double>& dist_coeffs, const o3d::camera::PinholeCameraIntrinsic& intr)
    {
        std::vector<cv::Point2f> inpoints, outpoints;
        for (const auto& f : features)
            inpoints.emplace_back(cv::Point2f(f.x(), f.y()));

        cv::Mat cv_intr, cv_dist;
        cv::eigen2cv(intr.intrinsic_matrix_, cv_intr);
        
        cv::undistortPoints(inpoints, outpoints, cv_intr, dist_coeffs);

        std::vector<Eigen::Vector2d> return_points;
        for (const auto& p : outpoints)
            return_points.emplace_back(Eigen::Vector2d(p.x, p.y));

        return return_points;
    }

    Eigen::Vector2d undistort_feature(const Eigen::Vector2d& feature, const std::vector<double>& dist_coeffs, const o3d::camera::PinholeCameraIntrinsic& intr)
    {
        return undistort_features({feature}, dist_coeffs, intr)[0];
    }

    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_features(const std::shared_ptr<o3d::geometry::Image> rgb, 
        const uint32_t max_feature_count, const o3d::camera::PinholeCameraIntrinsic* intr, const std::vector<double>* dist_coeffs)
    {
        Timer t;

        // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        // akaze->setThreshold(AKAZE_THRESHOLD);
        // cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(max_feature_count);
        cv::Mat descriptor;
        std::vector<cv::KeyPoint> keypoints;

        /**
         *  TODO: optimize this, divide to 4 quadrants and multithread?
         */
        cv::Mat cv_img = o3d_image_to_cv_image(rgb);
        /** NOTE: grayscaling the image saves a couple of milliseconds */
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);

        #ifdef DEBUG_VERBOSE
        t.stop("converted o3d image to cv::Mat");
        #endif

        if (intr != nullptr && dist_coeffs != nullptr && *dist_coeffs != ZERO_DISTORTION)
        {
            cv::Mat undistr_img;
            cv::Mat cv_intr;
            cv::eigen2cv(intr->intrinsic_matrix_, cv_intr);
            cv::undistort(cv_img, undistr_img, cv_intr, *dist_coeffs);
            cv_img = undistr_img;
            
            // cv::imshow("img", cv_img);
            // cv::waitKey(0);
        }
        
        #ifdef DEBUG_VERBOSE
        t.stop("undistort");
        #endif

        /**
         * NOTE: the tiled version of feature detection,
         *       ~3x faster but doesn't detect around cell borders.
         */
        // const int cell_width = cv_img.cols / 4;
        // const int cell_height = cv_img.rows / 4;
        // const auto [image_cells, cell_coordinates] = image_to_cells(cv_img, cell_width, cell_height);
        // std::vector<cv::Mat> cell_descriptors (image_cells.size(), cv::Mat());
        // std::vector<std::vector<cv::KeyPoint>> cell_keypoints (image_cells.size(), std::vector<cv::KeyPoint>());

        // // encapsulate the opencv function call in a lambda for convenience
        // auto f = [&](const cv::Mat cell, const int i) {
        //     detector->detectAndCompute(cell, cv::noArray(), cell_keypoints[i], cell_descriptors[i]);
        // };

        // std::vector<std::thread> cell_threads;
        // for (int i = 0; i < image_cells.size(); i++)
        // {
        //     cell_threads.emplace_back(std::thread(f, image_cells[i], i));
        // }

        // for (auto& t : cell_threads)
        //     t.join();

        // /**
        //  * NOTE: do descriptors have any location data?
        //  */
        // keypoints = cell_keypoints[0];
        // descriptor = cell_descriptors[0];
        // for (int i = 1; i < cell_keypoints.size(); i++)
        // {
        //     std::vector<cv::KeyPoint> kps = cell_keypoints[i];
        //     for (cv::KeyPoint& kp : kps)
        //     {
        //         kp.pt.x += cell_coordinates[i].x();
        //         kp.pt.y += cell_coordinates[i].y();
        //     }

        //     keypoints.insert(keypoints.end(), kps.begin(), kps.end());
        //     cv::vconcat(descriptor, cell_descriptors[i], descriptor);
        // }

        /**
         * TODO: test if detector->detect() can be tiled, if yes then
         *       detector->compute() can be tiled as well
         */
        detector->detectAndCompute(cv_img, cv::noArray(), keypoints, descriptor);
        
        #ifdef DEBUG_VERBOSE
        t.stop("feature detect and compute");
        #endif

        // cv::Mat out_img;
        // cv::drawKeypoints(cv_img, keypoints, out_img);
        // cv::imshow("it", out_img);
        // cv::waitKey(0);


        return std::make_tuple(keypoints, descriptor);
    }

    std::tuple<std::vector<cv::Mat>, std::vector<Eigen::Vector2i>> image_to_cells(const cv::Mat image, const int cell_size_x, const int cell_size_y)
    {
        std::vector<cv::Mat> image_cells;
        std::vector<Eigen::Vector2i> cell_coordinates;

        const int height = image.rows;
        const int width = image.cols;

        for (int y = 0; y <= height - cell_size_y; y += cell_size_y)
        {
            for (int x = 0; x <= width - cell_size_x; x += cell_size_x)
            {
                const cv::Rect grid_rect(x, y, cell_size_x, cell_size_y);
                image_cells.emplace_back(image(grid_rect));
                cell_coordinates.emplace_back(Eigen::Vector2i(x, y));
            }
        }

        return std::make_tuple(image_cells, cell_coordinates);
    }

    std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_akaze_features(const std::shared_ptr<cv::Mat> rgb, const o3d::camera::PinholeCameraIntrinsic* intr, const std::vector<double>* dist_coeffs)
    {
        // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        // akaze->setThreshold(AKAZE_THRESHOLD);
        cv::Ptr<cv::ORB> akaze = cv::ORB::create(1000);
        cv::Mat descriptor, cv_img;
        cv_img = *rgb;
        std::vector<cv::KeyPoint> keypoints;

        if (intr != nullptr && dist_coeffs != nullptr && *dist_coeffs != ZERO_DISTORTION)
        {
            cv::Mat undistr_img;
            cv::Mat cv_intr;
            cv::eigen2cv(intr->intrinsic_matrix_, cv_intr);
            cv::undistort(cv_img, undistr_img, cv_intr, *dist_coeffs);
            cv_img = undistr_img;
            // cv::imshow("img", cv_img);
            // cv::waitKey(0);
        }

        /** NOTE: grayscaling the image shaves ~10ms from the time, 88ms -> 80ms. */
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
        akaze->detectAndCompute(cv_img, cv::noArray(), keypoints, descriptor);

        // cv::Mat out_img;
        // cv::drawKeypoints(cv_img, keypoints, out_img);
        // cv::imshow("it", out_img);
        // cv::waitKey(0);

        return std::make_tuple(keypoints, descriptor);
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filter_matches_triangulation(Camera& cam0, Camera& cam1, const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> rc_corr)
    {
        const size_t CORR_SIZE = rc_corr.first.size();

        std::vector<cv::KeyPoint> kpf0 = cam0.kp_features;
        std::vector<cv::KeyPoint> kpf1 = cam1.kp_features;

        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filtered_corr;
        filtered_corr.first.reserve(CORR_SIZE);
        filtered_corr.second.reserve(CORR_SIZE);

        std::vector<cv::Point2f> kpf0_pt, kpf1_pt;
        kpf0_pt.reserve(CORR_SIZE);
        kpf1_pt.reserve(CORR_SIZE);

        cv::Mat tr_4d, P0, P1;
        cv::eigen2cv(cam0.P, P0);
        cv::eigen2cv(cam1.P, P1);

        // std::vector<Eigen::Vector3d> tr_3d;
        // tr_3d.reserve(kpf0.size());

        for (int i = 0; i < CORR_SIZE; i++)
        {
            kpf0_pt.emplace_back(kpf0[rc_corr.first[i]].pt);
            kpf1_pt.emplace_back(kpf1[rc_corr.second[i]].pt);
        }

        cv::triangulatePoints(P0, P1, kpf0_pt, kpf1_pt, tr_4d);

        for (int i = 0; i < CORR_SIZE; i++)
        {
            const Eigen::Vector3d cam0_view = cam0.rotation * CAMERA_FORWARD;
            const Eigen::Vector3d cam1_view = cam1.rotation * CAMERA_FORWARD;

            const Eigen::Vector3d p3d (
                    tr_4d.at<float>(0, 0) / tr_4d.at<float>(3, 0),
                    tr_4d.at<float>(1, 0) / tr_4d.at<float>(3, 0),
                    tr_4d.at<float>(2, 0) / tr_4d.at<float>(3, 0)
                );

            if (cam0_view.dot(p3d - cam0.position) < 0.0 || cam1_view.dot(p3d - cam1.position) < 0.0)
                continue;

            // tr_3d.emplace_back(p3d);
            filtered_corr.first.emplace_back(rc_corr.first[i]);
            filtered_corr.second.emplace_back(rc_corr.second[i]);
        }

        std::cout << "filtered " << std::to_string(rc_corr.first.size() - filtered_corr.first.size()) << " correspondences\n";
        return filtered_corr;
    }

    void filter_triangulated_3view(std::vector<Eigen::Vector3d>& trpoints, std::vector<std::vector<feature_vertex>>& fvs,
        const std::vector<Camera*>& cameras)
    {
        const std::vector<Eigen::Vector3d> points3d = trpoints;
        const std::vector<std::vector<feature_vertex>> ftracks = fvs;

        std::vector<Eigen::Vector3d> ret3d;
        ret3d.reserve(points3d.size());
        std::vector<std::vector<feature_vertex>> rett;

        const Eigen::Vector3d cam0_view = cameras.at(0)->rotation * CAMERA_FORWARD;
        const Eigen::Vector3d cam1_view = cameras.at(1)->rotation * CAMERA_FORWARD;
        const Eigen::Vector3d cam2_view = cameras.at(2)->rotation * CAMERA_FORWARD;

        const Eigen::Vector3d cam0_pos = cameras.at(0)->position;
        const Eigen::Vector3d cam1_pos = cameras.at(1)->position;
        const Eigen::Vector3d cam2_pos = cameras.at(2)->position;

        for (int i = 0; i < points3d.size(); i++)
        {
            const Eigen::Vector3d p3d = points3d[i];

            if (cam0_view.dot(p3d - cam0_pos) < 0.0 || cam1_view.dot(p3d - cam1_pos) < 0.0 || cam2_view.dot(p3d - cam2_pos) < 0.0)
                continue;

            ret3d.emplace_back(p3d);
            rett.emplace_back(ftracks[i]);
        }

        std::cout << "filtered out " << std::to_string(points3d.size() - ret3d.size()) << " points\n";

        trpoints = ret3d;
        fvs = rett;
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> match_features_no_homography(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& desc1, const std::vector<cv::KeyPoint>& kp2, const cv::Mat& desc2,
        const std::shared_ptr<o3d::geometry::Image> img1, const std::shared_ptr<o3d::geometry::Image> img2)
    {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> nn_matches;

        matcher.knnMatch(desc1, desc2, nn_matches, 2);

        // std::vector<cv::KeyPoint> matched1, matched2;
        std::vector<cv::KeyPoint> inliers1, inliers2;
        std::vector<uint32_t> inliers1_i, inliers2_i;
        std::vector<cv::DMatch> good_matches;

        int l_i = 0;

        for(size_t i = 0; i < nn_matches.size(); i++)
        {
            const cv::DMatch first = nn_matches[i][0];
            const float dist1 = nn_matches[i][0].distance;
            const float dist2 = nn_matches[i][1].distance;

            if (dist1 < NN_MATCH_RATIO * dist2)
            {
                inliers1.emplace_back(kp1[first.queryIdx]);
                inliers2.emplace_back(kp2[first.trainIdx]);

                inliers1_i.emplace_back(first.queryIdx);
                inliers2_i.emplace_back(first.trainIdx);

                good_matches.emplace_back(cv::DMatch(l_i, l_i, 0));
                l_i++;
            }
        }

        // debug draw matches if images are supplied
        if (img1 != nullptr && img2 != nullptr)
        {
            cv::Mat matchimg;
            cv::drawMatches(o3d_image_to_cv_image(img1), inliers1, o3d_image_to_cv_image(img2), inliers2, good_matches, matchimg);
            cv::imshow("img", matchimg);
            cv::waitKey(50);
        }

        return std::make_pair(inliers1_i, inliers2_i);
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> match_features(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& desc1, const std::vector<cv::KeyPoint>& kp2, const cv::Mat& desc2,
        const std::shared_ptr<o3d::geometry::Image> img1, const std::shared_ptr<o3d::geometry::Image> img2)
    {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> nn_matches;

        matcher.knnMatch(desc1, desc2, nn_matches, 2);

        std::vector<cv::KeyPoint> matched1, matched2;
        std::vector<uint32_t> matched1_i, matched2_i;
        cv::Mat inlier_mask, homography;

        for (unsigned int i = 0; i < nn_matches.size(); i++)
        {
            if (nn_matches[i][0].distance < NN_MATCH_RATIO * nn_matches[i][1].distance)
            {
                matched1.push_back(kp1[nn_matches[i][0].queryIdx]);
                matched2.push_back(kp2[nn_matches[i][0].trainIdx]);

                matched1_i.emplace_back(nn_matches[i][0].queryIdx);
                matched2_i.emplace_back(nn_matches[i][0].trainIdx);
            }
        }

        std::vector<cv::DMatch> inlier_matches;
        if (matched1.size() >= 4)
        {
            homography = findHomography(matches_to_points(matched1), matches_to_points(matched2), cv::RANSAC, ransac_thresh, inlier_mask);
            // std::cout << "homography:\n" << homography << "\n";
        }
        else
            throw std::runtime_error("not enough matched points, aborting in match_features!");

        std::vector<cv::DMatch> good_matches;
        std::vector<cv::KeyPoint> inliers1, inliers2;
        std::vector<uint32_t> inliers1_i, inliers2_i;
        std::map<size_t, size_t> ret_matches;

        for (size_t i = 0; i < matched1.size(); i++)
        {
            cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = matched1[i].pt.x;
            col.at<double>(1) = matched1[i].pt.y;
            col = homography * col;
            col /= col.at<double>(2);
            const double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) + pow(col.at<double>(1) - matched2[i].pt.y, 2));
            if (dist < INLIER_THRESHOLD)
            {
                // if (abs(matched1[i].pt.x - matched2[i].pt.x) > MAX_FEATURE_ABS_DEVIATION || abs(matched1[i].pt.y - matched2[i].pt.y) > MAX_FEATURE_ABS_DEVIATION)
                // {
                //     // this branch should never be hit, but things seem sometimes a bit unrealiable
                //     std::cout << "invalid point: " << matched1[i].pt.x << ", " << matched1[i].pt.y << " - " << matched2[i].pt.x << ", " << matched2[i].pt.y << "\n";
                //     continue;
                // }
                
                const int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);

                inliers1_i.emplace_back(matched1_i[i]);
                inliers2_i.emplace_back(matched2_i[i]);

                good_matches.push_back(cv::DMatch(new_i, new_i, 0));
            }
        }

        // debug draw matches if images are supplied
        if (img1 != nullptr && img2 != nullptr)
        {
            cv::Mat matchimg;
            cv::drawMatches(o3d_image_to_cv_image(img1), inliers1, o3d_image_to_cv_image(img2), inliers2, good_matches, matchimg);
            cv::imshow("img", matchimg);
            cv::waitKey(0);
        }


        // std::map<size_t, size_t> ret_matches;

        // for (size_t i = 0; i < nn_matches.size(); i++)
        // {
        //     const cv::DMatch first = nn_matches[i][0];
        //     const float dist1 = nn_matches[i][0].distance;
        //     const float dist2 = nn_matches[i][1].distance;

        //     if (dist1 < nn_match_ratio * dist2)
        //     {
        //         // matched1.push_back(kp1[first.queryIdx]);
        //         // matched2.push_back(kp2[first.trainIdx]);
        //         ret_matches[first.queryIdx] = first.trainIdx;
        //     }
        // }

        // for (const cv::DMatch& m : good_matches)
        // {
        //     ret_matches[m.queryIdx] = m.trainIdx;
        // }

        return std::make_pair(inliers1_i, inliers2_i);
    }

    std::vector<cv::Point2f> matches_to_points(const std::vector<cv::KeyPoint>& matches)
    {
        std::vector<cv::Point2f> points;
        points.reserve(matches.size());

        for (const cv::KeyPoint& kp : matches)
            points.emplace_back(kp.pt);

        return points;
    }

    std::vector<cv::Point2f> normalize_features(std::vector<cv::Point2f> points_vec)
    {
        // Averaging
        const double count = (double) points_vec.size();
        double xAvg = 0;
        double yAvg = 0;
        for (auto& member : points_vec)
        {
            xAvg = xAvg + member.x;
            yAvg = yAvg + member.y;
        }
        xAvg = xAvg / count;
        yAvg = yAvg / count;

        // Normalization
        std::vector<cv::Point2f> points3d;
        std::vector<double> distances;
        for (auto& member : points_vec)
        {

            double distance = (std::sqrt(std::pow((member.x - xAvg), 2) + std::pow((member.y - yAvg), 2)));
            distances.push_back(distance);
        }
        const double xy_norm = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

        // Create a matrix transforming the points into having mean (0,0) and mean distance to the center equal to sqrt(2)
        cv::Mat_<double> norm_mat(3, 3); 
        double diagonal_element = sqrt(2) / xy_norm;
        double element_13 = -sqrt(2) * xAvg / xy_norm;
        double element_23 = -sqrt(2)* yAvg/ xy_norm;

        norm_mat << diagonal_element, 0, element_13,
            0, diagonal_element, element_23,
            0, 0, 1;

        // Multiply the original points with the normalization matrix
        for (const auto& member : points_vec)
        {
            const cv::Mat triplet = (cv::Mat_<double>(3, 1) << member.x, member.y, 1);
            const cv::Mat temp = norm_mat * triplet;

            points3d.emplace_back(cv::Point2f(temp.at<float>(0), temp.at<float>(1)));
        }
        return points3d;
    }

    void draw_matches_indexes(const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1)
    {
        std::vector<cv::KeyPoint> matches0, matches1;
        matches0.reserve(matches.first.size());
        matches1.reserve(matches.second.size());

        cv::Mat matchimg;
        cv::hconcat(o3d_image_to_cv_image(cam0.rgb), o3d_image_to_cv_image(cam1.rgb), matchimg);
        cv::cvtColor(matchimg, matchimg, cv::COLOR_RGB2BGR);

        const int width = cam0.rgb->width_;

        srand(time(NULL));

        float h_i = 0.0f;
        const float increment = 360.0f / (float)matches.first.size();
        for (int i = 0; i < matches.first.size(); i++)
        {
            const cv::Point2f kp0 = cam0.kp_features.at(matches.first.at(i)).pt;
            const cv::Point2f kp1 = cam1.kp_features.at(matches.second.at(i)).pt;

            const auto [r, g, b] = stitcher3d::rgbd::HSL_to_RGB(h_i, 90.0f + (float)rand() / (float)RAND_MAX * 10.0f, 50.0f + (float)rand() / (float)RAND_MAX * 10.0f);
            const cv::Scalar rand_color (r, g, b);
            h_i += increment;

            cv::circle(matchimg, cv::Point(kp0.x, kp0.y), 3, rand_color);
            cv::circle(matchimg, cv::Point(kp1.x + width, kp1.y), 3, rand_color);

            cv::line(matchimg, cv::Point(kp0.x, kp0.y), cv::Point(kp1.x + width, kp1.y), rand_color, 1);
        }

        static int wait_time = 1;
        // if (matches.first.size() < 20)
        //     wait_time = 0;

        cv::imshow("img", matchimg);
        cv::waitKey(wait_time);
    }

    std::vector<std::pair<uint32_t, uint32_t>> filter_matches_radius(
        const std::vector<std::pair<uint32_t, uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1,
        const float max_radius_ratio, const float min_radius_ratio)
    {
        std::vector<std::pair<uint32_t, uint32_t>> filtered;
        filtered.reserve(matches.size());

        const float max_x_delta = cam0.rgb->width_ * max_radius_ratio;
        const float max_y_delta = cam0.rgb->height_ * max_radius_ratio;

        const float min_x_delta = cam0.rgb->width_ * min_radius_ratio;
        const float min_y_delta = cam0.rgb->height_ * min_radius_ratio;

        for (int i = 0; i < matches.size(); i++)
        {
            const cv::Point2f kp0 = cam0.kp_features.at(matches.at(i).first).pt;
            const cv::Point2f kp1 = cam1.kp_features.at(matches.at(i).second).pt;

            const double x_diff_abs = abs((double)kp0.x - (double)kp1.x);
            const double y_diff_abs = abs((double)kp0.y - (double)kp1.y);

            if (FEATURE_FILTER_CHECK_MIN)
            {
                if (x_diff_abs < max_x_delta && x_diff_abs > min_x_delta &&
                    y_diff_abs < max_y_delta && y_diff_abs > min_y_delta)
                {
                    filtered.emplace_back(std::make_pair(matches.at(i).first, matches.at(i).second));
                }
            }
            else
            {
                if (x_diff_abs < max_x_delta && y_diff_abs < max_y_delta)
                {
                    filtered.emplace_back(std::make_pair(matches.at(i).first, matches.at(i).second));
                }
            }
        }

        filtered.shrink_to_fit();
        return filtered;
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filter_matches_radius(
        const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matches, 
        const Camera& cam0, const Camera& cam1, const float max_radius_ratio, 
        const float min_radius_ratio)
    {
        if (matches.first.size() != matches.second.size())
            throw std::runtime_error("matches sizes didn't match, aborting in filter_matches_radius!");

        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> filtered;
        filtered.first.reserve(matches.first.size());
        filtered.second.reserve(matches.second.size());

        const float max_x_delta = cam0.rgb->width_ * max_radius_ratio;
        const float max_y_delta = cam0.rgb->height_ * max_radius_ratio;

        const float min_x_delta = cam0.rgb->width_ * min_radius_ratio;
        const float min_y_delta = cam0.rgb->height_ * min_radius_ratio;

        for (int i = 0; i < matches.first.size(); i++)
        {
            const cv::Point2f kp0 = cam0.kp_features.at(matches.first.at(i)).pt;
            const cv::Point2f kp1 = cam1.kp_features.at(matches.second.at(i)).pt;

            const double x_diff_abs = abs((double)kp0.x - (double)kp1.x);
            const double y_diff_abs = abs((double)kp0.y - (double)kp1.y);

            if (FEATURE_FILTER_CHECK_MIN)
            {
                if (x_diff_abs < max_x_delta && x_diff_abs > min_x_delta &&
                    y_diff_abs < max_y_delta && y_diff_abs > min_y_delta)
                {
                    filtered.first.emplace_back(matches.first.at(i));
                    filtered.second.emplace_back(matches.second.at(i));
                }
            }
            else
            {
                if (x_diff_abs < max_x_delta && y_diff_abs < max_y_delta)
                {
                    filtered.first.emplace_back(matches.first.at(i));
                    filtered.second.emplace_back(matches.second.at(i));
                }
            }
        }

        filtered.first.shrink_to_fit();
        filtered.second.shrink_to_fit();

        return filtered;
    }

}

