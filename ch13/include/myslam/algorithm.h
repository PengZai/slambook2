//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 * @equation description: s1 * x1 = P * X, s1 is depth scale, x1 is normalized pixel in camera coordinate, P is [R|t], X is world point(actuall point in left camera coordinate)
 * 
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);      // u * P3 - P1
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);  // v * P3 - P2
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    // if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
    //     // solution qualtiy is not good, give up
    //     return true;
    // }

    // return false;
    std::cout << "condition number : " << svd.singularValues()[0] / svd.singularValues()[3] << std::endl;

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // solution qualtiy is not good, give up
        return true;
    }

    return false;
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
