#version 460

#define DEPTH_SCALE 0.001f
#define DEPTH_CLIP_FAR 6.0f
#define SMALL_FLOAT 0.00000001f
#define INVALID_POINT vec3(0.f, 0.f, 0.f)

#define ORIGIN vec3(0.f, 0.f, 0.f)

#define WIDTH 1280
#define HEIGHT 720
#define CX 637.74400638
#define CY 358.96757428
#define FX 912.91984103
#define FY 912.39513184

struct pixel_data
{
    float p_x;
    float p_y;
    float p_z;

    int tr_i;

    int coord_x;
    int coord_y;

    float padding0;
    float padding1;
};

struct vertex_data
{
    float v_x;
    float v_y;
    float v_z;
    float padding;
};

layout (local_size_x = 1024) in;

uniform sampler2D depth_texture;
uniform int triangle_count;
uniform mat4 camera_transform;
uniform float depth_cloud_world_scale;

layout (std430, binding = 0) buffer out_buffer_hit_data
{
    pixel_data pdata[];
} out_data;

layout (std430, binding = 1) buffer in_buffer_triangle_data
{
    vertex_data vdata[];
} in_data;

// layout (std430, binding = 2) buffer in_depth
// {
//     float d[1280*720];
// } in_d;

vec3 pixel_to_world(const vec2 pixel_coord, const float d_val)
{
    vec4 world_point = vec4(
        (pixel_coord.x - CX) * d_val / FX,
        (pixel_coord.y - CY) * d_val / FY,
        d_val,
        1.f
    );

    mat4 flip_mat;
    flip_mat[0] = vec4(1.f, 0.f, 0.f, 0.f);
    flip_mat[1] = vec4(0.f, -1.f, 0.f, 0.f);
    flip_mat[2] = vec4(0.f, 0.f, -1.f, 0.f);
    flip_mat[3] = vec4(0.f, 0.f, 0.f, 1.f);

    return (flip_mat * world_point).xyz;
}

bool is_close_f(const float val1, const float val2)
{
    if (abs(val1 - val2) < SMALL_FLOAT)
        return true;

    return false;
}

vec3 intersect_triangle_ray(const vec3 origin, const vec3 direction, const vec3 v0, const vec3 v1, const vec3 v2)
{
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v1;
    vec3 e2 = v0 - v2;
    const vec3 tr_normal = normalize(cross(v1 - v0, v2 - v0));
    float n_dot_dir = dot(tr_normal, direction);

    float d = dot(tr_normal, v0);
    float t = (dot(tr_normal, origin) + d) / n_dot_dir;

    if (t < 0.f)
        return INVALID_POINT;

    const vec3 point = origin + (t * direction);
    vec3 perpendicular;


    vec3 vp0 = point - v0;
    perpendicular = cross(e0, vp0);

    if (dot(tr_normal, perpendicular) < 0.f)
        return INVALID_POINT;


    vec3 vp1 = point - v1;
    perpendicular = cross(e1, vp1);

    if (dot(tr_normal, perpendicular) < 0.f)
        return INVALID_POINT;


    vec3 vp2 = point - v2;
    perpendicular = cross(e2, vp2);

    if (dot(tr_normal, perpendicular) < 0.f)
        return INVALID_POINT;

    return point;
}

void set_false(ivec2 xy_coord, int linear_out_coord)
{
    out_data.pdata[linear_out_coord].p_x = 0.f;
    out_data.pdata[linear_out_coord].p_y = 0.f;
    out_data.pdata[linear_out_coord].p_z = 0.f;

    out_data.pdata[linear_out_coord].tr_i = -1;
    // out_data.pdata[linear_out_coord].coord_x = xy_coord.x;
    out_data.pdata[linear_out_coord].coord_x = xy_coord.x;
    out_data.pdata[linear_out_coord].coord_y = xy_coord.y;

    out_data.pdata[linear_out_coord].padding0 = 0;
    out_data.pdata[linear_out_coord].padding1 = 0;
}

void DEBUG_set_override(ivec2 xy_coord, int linear_out_coord, float value)
{
    out_data.pdata[linear_out_coord].p_x = value;
    out_data.pdata[linear_out_coord].p_y = 0.f;
    out_data.pdata[linear_out_coord].p_z = 0.f;

    out_data.pdata[linear_out_coord].tr_i = -4;
    // out_data.pdata[linear_out_coord].coord_x = xy_coord.x;
    out_data.pdata[linear_out_coord].coord_x = xy_coord.x;
    out_data.pdata[linear_out_coord].coord_y = xy_coord.y;

    out_data.pdata[linear_out_coord].padding0 = 0;
    out_data.pdata[linear_out_coord].padding1 = 0;
}

void main()
{
    const int linear_out_coord = int(gl_GlobalInvocationID.x);
    // const int linear_out_coord = int(gl_GlobalInvocationID.y) * WIDTH + int(gl_GlobalInvocationID.x);

    const ivec2 xy_coord = ivec2(
        int(linear_out_coord % WIDTH),
        int(linear_out_coord / WIDTH)
    );
    // const ivec2 xy_coord = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

    // set_false(xy_coord, linear_out_coord);
    // return;

    float d_val = texture2D(depth_texture, vec2(float(xy_coord.x) / float(WIDTH), float(xy_coord.y) / float(HEIGHT))).r;

    if (abs(d_val) < 0.001f)
    {
        set_false(xy_coord, linear_out_coord);
        return;
    }

    vec3 v0;
    vec3 v1;
    vec3 v2;
    // world point is in camera space
    // const vec3 world_point = (camera_transform * vec4(pixel_to_world(xy_coord, d_val), 1.f)).xyz;
    const vec3 world_point = pixel_to_world(xy_coord, d_val);
    float smallest_length = 100000000.f;
    float len;
    vec3 smallest_point = INVALID_POINT;
    int smallest_tr_i = -2;
    int linear_tr_i = 0;

    const mat4 camera_transform_inverse = inverse(camera_transform);

    // const vec3 ray_origin = (camera_transform_inverse * vec4(ORIGIN, 1.f)).xyz;
    // const vec3 ray_direction = normalize((camera_transform_inverse * vec4(world_point, 1.f)).xyz - ray_origin);
    // const vec3 ray_direction = normalize((camera_transform * vec4(world_point, 1.f)).xyz - ray_origin);
    // const vec3 ray_origin = (camera_transform * vec4(ORIGIN, 1.f)).xyz;
    const vec3 ray_origin = ORIGIN;

    // ray_direction is in camera space
    const vec3 ray_direction = normalize(world_point - ray_origin);
    
    if (triangle_count > 0)
    {
        for (int current_tr_i = 0; current_tr_i < triangle_count * 3; current_tr_i+=3)
        {
            v0 = vec3(
                in_data.vdata[current_tr_i].v_x,
                in_data.vdata[current_tr_i].v_y,
                in_data.vdata[current_tr_i].v_z
            );

            v1 = vec3(
                in_data.vdata[current_tr_i + 1].v_x,
                in_data.vdata[current_tr_i + 1].v_y,
                in_data.vdata[current_tr_i + 1].v_z
            );

            v2 = vec3(
                in_data.vdata[current_tr_i + 2].v_x,
                in_data.vdata[current_tr_i + 2].v_y,
                in_data.vdata[current_tr_i + 2].v_z
            );

            // project the triangle into camera's space from world space
            v0 = (camera_transform_inverse * vec4(v0, 1.f)).xyz;
            v1 = (camera_transform_inverse * vec4(v1, 1.f)).xyz;
            v2 = (camera_transform_inverse * vec4(v2, 1.f)).xyz;

            // point is in camera space
            const vec3 point = intersect_triangle_ray(ray_origin, ray_direction, v0, v1, v2);
            
            // if the intersected point is nowhere near the required pointcloud point 
            // --> discard
            // must be done because inaccuracies in the mesh, which leads to near-misses
            // in the ray projection
            // if (point != INVALID_POINT && length(point - world_point) < 1.5)

            // const vec3 wpoint_t = (camera_transform_inverse * vec4(world_point, 0.f)).xyz;
            // const vec3 point_t = (camera_transform_inverse * vec4(point, 0.f)).xyz;

            // find a way to optimize the distance breaking threshold
            if (point != INVALID_POINT && abs((length(point) / depth_cloud_world_scale) - d_val) < 0.45)
            {
                len = length(point - ray_origin);
                if (len < smallest_length)
                {
                    smallest_length = len;
                    smallest_point = point;
                    smallest_tr_i = linear_tr_i;

                    // break;
                }
            }
            linear_tr_i++;
        }
    }

    // const float dist = length(smallest_point);
    smallest_point = (camera_transform * vec4(smallest_point, 1.f)).xyz;

    out_data.pdata[linear_out_coord].p_x = float(smallest_point.x);
    out_data.pdata[linear_out_coord].p_y = float(smallest_point.y);
    out_data.pdata[linear_out_coord].p_z = float(smallest_point.z);

    out_data.pdata[linear_out_coord].tr_i = smallest_tr_i;
    out_data.pdata[linear_out_coord].coord_x = xy_coord.x;
    out_data.pdata[linear_out_coord].coord_y = xy_coord.y;

    out_data.pdata[linear_out_coord].padding0 = 0.f; //d_val;
    out_data.pdata[linear_out_coord].padding1 = 0.f; //dist;

}