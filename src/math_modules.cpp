#include "math_modules.h"


namespace stitcher3d
{
namespace math
{

void triangulate_dlt(const Mat34 &P1, const Eigen::Vector2d &x1,
                    const Mat34 &P2, const Eigen::Vector2d &x2,
                    Eigen::Vector4d *X_homogeneous)
{
    Eigen::Matrix4d design;
    for (int i = 0; i < 4; ++i)
    {
        design(0,i) = x1(0) * P1(2,i) - P1(0,i);
        design(1,i) = x1(1) * P1(2,i) - P1(1,i);
        design(2,i) = x2(0) * P2(2,i) - P2(0,i);
        design(3,i) = x2(1) * P2(2,i) - P2(1,i);
    }
    compute_nullspace(&design, X_homogeneous);
}

void triangulate_dlt(const Mat34 &P1, const Eigen::Vector2d &x1,
                    const Mat34 &P2, const Eigen::Vector2d &x2,
                    Eigen::Vector3d *X_euclidean)
{
    Eigen::Vector4d X_homogeneous;
    triangulate_dlt(P1, x1, P2, x2, &X_homogeneous);
    homogeneous_to_euclidean(X_homogeneous, X_euclidean);
}

void homogeneous_to_euclidean(const Eigen::Vector4d& H, Eigen::Vector3d* X)
{
    double w = H(3);
    *X << H(0) / w, H(1) / w, H(2) / w;
}

void P_From_KRt(const Eigen::Matrix3d& intr, const Eigen::Matrix3d& rotation, 
    const Eigen::Vector3d& translation, Mat34 *P)
{
    P->block<3, 3>(0, 0) = rotation;
    P->col(3) = translation;
    (*P) = intr * (*P);
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> solution_in_front_of_camera(const std::vector<Eigen::Matrix3d>& rotations, 
    const std::vector<Eigen::Vector3d>& translations, const Eigen::Matrix3d& intr1, const Eigen::Matrix3d& intr2,
    const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
    if (rotations.size() != translations.size() || rotations.size() != 4)
        throw std::runtime_error("rotations and translations size didn't match or weren't 4, aborting in solution_in_front_of_camera!");

    Mat34 P1, P2;
    Eigen::Matrix3d R1;
    Eigen::Vector3d t1;

    R1.setIdentity();
    t1.setZero();

    P_From_KRt(intr1, R1, t1, &P1);

    for (int i = 0; i < 4; i++)
    {
        // const Eigen::Matrix3d &R2 = FLIP_TRANSFORM_3D_SPECIAL * rotations[i];
        // const Eigen::Vector3d &t2 = FLIP_TRANSFORM_3D * translations[i];
        const Eigen::Matrix3d &R2 = rotations[i];
        const Eigen::Vector3d &t2 = translations[i];

        P_From_KRt(intr2, R2, t2, &P2);
        Eigen::Vector3d X;
        triangulate_dlt(P1, x1, P2, x2, &X);
        double d1 = depth_from_RtX(R1, t1, X);
        double d2 = depth_from_RtX(R2, t2, X);

        if (d1 > 0 && d2 > 0)
            return std::make_tuple(rotations.at(i), translations.at(i), true);

    }
    
    return std::make_tuple(Eigen::Matrix3d(), Eigen::Vector3d(), false);
}

template <typename TMat, typename TVec>
double compute_nullspace(TMat *A, TVec *nullspace)
{
  Eigen::JacobiSVD<TMat> svd(*A, Eigen::ComputeFullV);
  (*nullspace) = svd.matrixV().col(A->cols()-1);

  if (A->rows() >= A->cols())
    return svd.singularValues()(A->cols()-1);
  else
    return 0.0;
}

std::tuple<Eigen::Matrix3d, double> compute_essential_8_point(const std::vector<Eigen::Vector2d>& x1,
    const std::vector<Eigen::Vector2d>& x2)
{
    if (x1.size() != x2.size() || x1.size() != 8)
        throw std::runtime_error("point vector sizes didn't match or weren't 8, aborting in compute_essential_8_point!");

    Eigen::Matrix<double, 2, 8> x1_M, x2_M;
    for (int i = 0; i < 8; i++)
    {
        x1_M(0, i) = x1[i][0];
        x1_M(1, i) = x1[i][1];

        x2_M(0, i) = x2[i][0];
        x2_M(1, i) = x2[i][1];
    }

    const int n = x1_M.cols();

    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; i++)
    {
        A(i, 0) = x2_M(0, i) * x1_M(0, i);
        A(i, 1) = x2_M(0, i) * x1_M(1, i);
        A(i, 2) = x2_M(0, i);
        A(i, 3) = x2_M(1, i) * x1_M(0, i);
        A(i, 4) = x2_M(1, i) * x1_M(1, i);
        A(i, 5) = x2_M(1, i);
        A(i, 6) = x1_M(0, i);
        A(i, 7) = x1_M(1, i);
        A(i, 8) = 1;
    }

    Eigen::Matrix<double, 9, 1> f;
    float smaller_singular_value = compute_nullspace(&A, &f);

    Eigen::Matrix3d E;
    E = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(f.data());

    return std::make_tuple(E, smaller_singular_value);
}

std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>> 
    r_and_t_from_essential_matrix(const Eigen::Matrix3d& E)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> USV(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = USV.matrixU();
    Eigen::Vector3d d = USV.singularValues();
    Eigen::Matrix3d Vt = USV.matrixV().transpose();

    if (U.determinant() < 0)
        U.col(2) *= -1;

    if (Vt.determinant() < 0)
        Vt.row(2) *= -1;

    Eigen::Matrix3d W;
    W <<    0, -1, 0,
            1, 0, 0,
            0, 0, 1;

    Eigen::Matrix3d U_W_Vt = U * W * Vt;
    Eigen::Matrix3d U_Wt_Vt = U * W.transpose() * Vt;

    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;

    Rs.resize(4);
    Rs[0] = U_W_Vt;
    Rs[1] = U_W_Vt;
    Rs[2] = U_Wt_Vt;
    Rs[3] = U_Wt_Vt;

    ts.resize(4);
    ts[0] = U.col(2);
    ts[1] = -U.col(2);
    ts[2] = U.col(2);
    ts[3] = -U.col(2);

    return std::make_tuple(Rs, ts);
}

Eigen::Quaterniond quaternion_look_rotation(const Eigen::Vector3d& forward, 
            const Eigen::Vector3d& up)
{
    double x, y, z, w;

    Eigen::Vector3d f_norm = forward.normalized();
    Eigen::Vector3d up_norm = up.normalized();
    Eigen::Vector3d right = up_norm.cross(f_norm);

    w = sqrt(1.f + right.x() + up.y() + forward.z()) * 0.5f;
    float w4_recip = 1.f / (4.f * w);

    x = (up.z() - forward.y()) * w4_recip;
    y = (forward.x() - right.z()) * w4_recip;
    z = (right.y() - up.x()) * w4_recip;

    Eigen::Quaterniond q(w, x, y, z);
    return q;
}

std::tuple<Eigen::Vector3d, bool> intersect_trianle_ray(const Eigen::Vector3d& origin,
                const Eigen::Vector3d& direction,
                const Eigen::Vector3i& triangle, const std::vector<Eigen::Vector3d>& vertices,
                const Eigen::Vector3d& tr_normal)
{
    const float n_dot_dir = tr_normal.dot(direction);
    if (is_close(n_dot_dir, 0.f))
        return std::make_tuple(origin, false);

    const Eigen::Vector3d v0 = vertices[triangle[0]];
    const Eigen::Vector3d v1 = vertices[triangle[1]];
    const Eigen::Vector3d v2 = vertices[triangle[2]];

    const float d = tr_normal.dot(v0);
    const float t = (tr_normal.dot(origin) + d) / n_dot_dir;
    // const float denom = tr_normal.dot(tr_normal);

    if (t < 0.f)
        return std::make_tuple(origin, false);

    const Eigen::Vector3d point = origin + t * direction;
    Eigen::Vector3d perpendicular;

    const Eigen::Vector3d e0 = v1 - v0;
    const Eigen::Vector3d vp0 = point - v0;
    perpendicular = e0.cross(vp0);

    if (tr_normal.dot(perpendicular) < 0.f)
        return std::make_tuple(origin, false);

    const Eigen::Vector3d e1 = v2 - v1;
    const Eigen::Vector3d vp1 = point - v1;
    perpendicular = e1.cross(vp1);

    if (tr_normal.dot(perpendicular) < 0.f)
        return std::make_tuple(origin, false);

    const Eigen::Vector3d e2 = v0 - v2;
    const Eigen::Vector3d vp2 = point - v2;
    perpendicular = e2.cross(vp2);

    if (tr_normal.dot(perpendicular) < 0.f)
        return std::make_tuple(origin, false);

    return std::make_tuple(point, true);
}

std::tuple<Eigen::Vector3d, bool, int> raycast_from_point_to_surface(const Eigen::Vector3d& origin,
                const Eigen::Vector3d& direction,
                const std::vector<Eigen::Vector3d>& vertices,
                const std::vector<Eigen::Vector3i>& triangles,
                const std::vector<Eigen::Vector3d>& triangle_normals)
{
    Eigen::Vector3d point;

    std::vector<float> hit_l;
    std::vector<Eigen::Vector3d> hit_points;
    std::vector<int> hit_tr_i;

    for (int i = 0; i < triangles.size(); i++)
    {
        float u, v;
        auto [point, hit] = intersect_trianle_ray(origin, direction.normalized(), triangles[i], vertices, triangle_normals[i]);

        if (hit)
        {
            hit_points.emplace_back(point);
            hit_l.emplace_back((point - origin).norm());
            hit_tr_i.push_back(i);

            // std::cout << "\nhit with point\n" << point << "\n compared to \n\n" 
            //     << origin + direction << "\n";
        }
    }

    if (hit_tr_i.size() == 0)
        return std::make_tuple(origin, false, -1);
    
    const float min_dist = *std::min_element(hit_l.begin(), hit_l.end());
    const auto min_dist_iter = std::find(hit_l.begin(), hit_l.end(), min_dist);
    const int min_dist_i = std::distance(hit_l.begin(), min_dist_iter);

    return std::make_tuple(hit_points[min_dist_i], true, hit_tr_i[min_dist_i]);
}

/*
inline float vector_angle(const Eigen::Vector3d& f, const Eigen::Vector3d& s)
{
    const float dot = f.dot(s);
    const float dot_f = f.dot(f);
    const float dot_s = s.dot(s);

    return acos(dot/sqrt(dot_f * dot_s));
}
*/

Eigen::Vector4d euclidean_to_homogenous(const Eigen::Vector3d eucl)
{
    return Eigen::Vector4d(eucl.x(), eucl.y(), eucl.z(), 1.0);
}

}
}
