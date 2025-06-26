#pragma once

#include "config.h"
#include <ceres/ceres.h>


bool bundleAdjustmentPoseOnlyCeres(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
);

class reprojectionCostFunctionForPoseOnly : public ceres::SizedCostFunction<
2, /* number of residuals, which 2D observations */
6  /* number of parameters for first parameter block, which is 6D pose parameters */
> {


public:

  reprojectionCostFunctionForPoseOnly(const Eigen::Vector3d &p_w, const Eigen::Vector2d &obs, const Eigen::Matrix<double, 3, 3> &K ) : obs_(obs), K_(K), p_w_(p_w){}

  virtual bool Evaluate(double const * const * parameters,
                        double* residuals,
                        double** jacobians) const {
      // se3 for pose
      // Eigen::Matrix<double, 6, 1> se3_pose;
      // se3_pose << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5];

      Eigen::Map<const Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> se3_pose(parameters[0]);
      // Eigen::Matrix<double, 1, 6> se3_pose(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
      // se3_pose << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5];

      SE3 T = SE3::exp(se3_pose);

      const Eigen::Vector3d p_c = T * p_w_;
      Eigen::Vector3d reproject_pixel = K_* p_c;

      const double X = p_c[0];
      const double Y = p_c[1];
      const double Z = p_c[2];
      const double inv_Z = 1.0 / ( Z + 1e-18 );
      const double inv_Z2 = inv_Z * inv_Z;
      const double fx = K_(0, 0);
      const double fy = K_(1, 1);

      reproject_pixel = reproject_pixel * inv_Z;

      // Eigen::Map<Eigen::Vector2d> reproject_res(residuals);
      // u - reproject_u
      // v - reproject_v
      // reproject_res = obs_ - reproject_pixel.head<2>(); 
      residuals[0] = obs_[0] - reproject_pixel[0];
      residuals[1] = obs_[1] - reproject_pixel[1];


      if(jacobians){

        const double J00 = -fx * inv_Z;
        const double J02 = fx * X * inv_Z2;
        const double J03 = fx * X * Y * inv_Z2;
        const double J04 = -fx - fx * X * X * inv_Z2;
        const double J05 = fx * Y * inv_Z;

        const double J11 = -fy * inv_Z;
        const double J12 = fy * Y * inv_Z2;
        const double J13 = fy + fy * Y * Y * inv_Z2;
        const double J14 = -fy * X * Y * inv_Z2;
        const double J15 = -fy * X * inv_Z;

        if(jacobians[0]){
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_res_by_pose(jacobians[0]);
          jacobian_res_by_pose << J00, 0  , J02, J03, J04, J05,
                                  0  , J11, J12, J13, J14, J15;
        }

      }



      return true;

  }

private:
  const Eigen::Vector3d p_w_;
  const Eigen::Vector2d obs_;
  const Eigen::Matrix<double, 3, 3> K_;

};

