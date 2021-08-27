#pragma once

#include <cstdint>
#include <iostream>

namespace stitcher3d::registration
{
	enum PipelineState
	{
	    Full,
	    PointcloudOnly,
	    MeshOnly,
	    ProjectOnly
	};
	
	struct RegistrationParams
	{
	    uint32_t candidate_chain_len;
	    uint32_t landmark_min_count;
	    bool refine_cameras;
	    uint32_t skip_stride;
	    double depth_far_clip;
	    uint32_t max_feature_count;
	    std::string assets_path;
	};
}