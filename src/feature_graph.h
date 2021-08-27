#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <Open3D/Open3D.h>

namespace stitcher3d
{

struct feature_vertex
{
    uint32_t camera_index;
    uint32_t feature_index;
};

struct feature_graph
{
    std::vector<std::vector<feature_vertex>> connected_vertices;

    void remove_camera(const uint32_t camid)
    {
        for (std::vector<feature_vertex>& track : connected_vertices)
        {
            std::vector<feature_vertex>::iterator it = track.begin();
            while (it != track.end())
            {
                if (it->camera_index == camid)
                    it = track.erase(it);
                else
                    it++;
            }
        }
    }

    std::vector<std::vector<feature_vertex>> connected_vertices_for_cameras(const std::vector<uint32_t> camera_ids) const
    {
        std::vector<std::vector<feature_vertex>> ret_track;

        for (const std::vector<feature_vertex>& track : connected_vertices)
        {
            std::vector<feature_vertex> fvtrack;
            for (const feature_vertex& fv : track)
            {
                for (const uint32_t cid : camera_ids)
                {
                    if (cid == fv.camera_index)
                        fvtrack.emplace_back(fv);
                }
            }
            if (fvtrack.size() == camera_ids.size())
                ret_track.emplace_back(fvtrack);
        }

        return ret_track;
    }

    void add_edge(const uint32_t camera_i_1, const uint32_t feature_i_1,
                  const uint32_t camera_i_2, const uint32_t feature_i_2)
    {
        // find and connect
        bool found = false;
        for (int i = 0; i < connected_vertices.size(); i++)
        {
            // check if the first vertex exists,
            // if it does -> append if not break and add a new track
            const auto edge_first_vertex = std::find_if(connected_vertices[i].begin(), connected_vertices[i].end(), [&camera_i_1, &feature_i_1](const feature_vertex& fv) {
                if (fv.camera_index == camera_i_1 && fv.feature_index == feature_i_1)
                {
                    return true;
                }
                return false;
            });

            // not found, continue searching
            if (edge_first_vertex == connected_vertices[i].end())
            {
                continue;
            }
            else
            {
                // track found, add the new vertex (if not already found)
                const auto edge_second_vertex = std::find_if(connected_vertices[i].begin(), connected_vertices[i].end(), [&camera_i_2, &feature_i_2](const feature_vertex& fv) {
                    if (fv.camera_index == camera_i_2 && fv.feature_index == feature_i_2)
                    {
                        return true;
                    }
                    return false;
                });
                if (edge_second_vertex != connected_vertices[i].end())
                    break;

                connected_vertices[i].emplace_back(feature_vertex { camera_i_2, feature_i_2 });
                found = true;
                break;
            }
        }
        
        // add a new track
        if (!found)
        {
            connected_vertices.emplace_back(std::vector<feature_vertex>{ 
                feature_vertex { camera_i_1, feature_i_1 }, feature_vertex { camera_i_2, feature_i_2 } });
        }
    }

    void remove_tracks_length_less_than(const uint32_t min_length)
    {
        std::vector<std::vector<feature_vertex>>::iterator it = connected_vertices.begin();
        while (it != connected_vertices.end())
        {
            if (it->size() < min_length)
                it = connected_vertices.erase(it);
            else
                it++;
        }
    }

    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> 
        get_3view_track(const uint32_t cam0_id, const uint32_t cam1_id, const uint32_t cam2_id) const
    {
        std::vector<uint32_t> track0, track1, track2;

        /**
         *  NOTE:
         *      loop through all tracks and find the ones which
         *      have all 3 camera_ids listed
         */

        for (const auto& track : connected_vertices)
        {
            const auto& fv0 = std::find_if(track.begin(), track.end(), [&cam0_id](const feature_vertex& fv) {
                if (fv.camera_index == cam0_id)
                    return true;
                return false;
            });
            if (fv0 == track.end())
                continue;
            
            const auto& fv1 = std::find_if(track.begin(), track.end(), [&cam1_id](const feature_vertex& fv) {
                if (fv.camera_index == cam1_id)
                    return true;
                return false;
            });
            if (fv1 == track.end())
                continue;

            const auto& fv2 = std::find_if(track.begin(), track.end(), [&cam2_id](const feature_vertex& fv) {
                if (fv.camera_index == cam2_id)
                    return true;
                return false;
            });
            if (fv2 == track.end())
                continue;

            // all 3 cameras were found in this track -> add to retvals
            track0.emplace_back(fv0->feature_index);
            track1.emplace_back(fv1->feature_index);
            track2.emplace_back(fv2->feature_index);
        }

        return std::make_tuple(track0, track1, track2);
    }

};

}