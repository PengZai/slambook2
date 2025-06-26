#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>  // 🔴 Needed to use cv::cv2eigen()
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>


#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/rotation.h>


using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


void bundleAdjustmentCeres(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
);

void bundleAdjustmentCeresAutoDiff(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
);

void bundleAdjustmentPoseOnlyCeres(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
);

void bundleAdjustmentPoseOnlyCeresAutoDiff(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
);


void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
  Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  Mat d1 = imread(argv[3], cv::IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch m:matches) {
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;


  t1 = chrono::steady_clock::now();
  Mat r2, tt2;
  cv::Mat inliers;
  bool success = solvePnPRansac(pts_3d, pts_2d, K, Mat(), r2, tt2, false, 100, 4.0, 0.99, inliers); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R2;
  cv::Rodrigues(r2, R2); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve solvePnPRansac in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R2 << endl;
  cout << "t=" << endl << tt2 << endl;

  t1 = chrono::steady_clock::now();
  Mat r3, tt3;

  std::vector<cv::Point2f> undistorted_points;
  cv::undistortPoints(pts_2d, undistorted_points, K, Mat());
  cv::Mat inliers3;
  bool success3 = solvePnPRansac(pts_3d, undistorted_points, Mat::eye(3,3,CV_64F), Mat(), r3, tt3, false, 100, 4.0, 0.99, inliers3); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R3;
  cv::Rodrigues(r3, R3); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve solvePnPRansac in normalized plane in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R3 << endl;
  cout << "t=" << endl << tt3 << endl;

  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  std::vector<Eigen::Vector3d> my_pts_3d_eigen;
  std::vector<Eigen::Vector2d> my_pts_2d_eigen;

  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    my_pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    my_pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    // std::cout << "pts_3d_eigen:" << pts_3d_eigen[i].x() << " , " <<  pts_3d_eigen[i].y() << " , " << pts_3d_eigen[i].z() << "| "
    // << "my_pts_3d_eigen:" << my_pts_3d_eigen[i].x() << " , " <<  my_pts_3d_eigen[i].y() << " , " << my_pts_3d_eigen[i].z()
    // << std::endl;
  }

  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;


  Eigen::Matrix<double, 3, 3> K_eigen;
  cv::cv2eigen(K, K_eigen);

  std::cout << "K" << K << std::endl;;
  std::cout << "K_eigen\n" << K_eigen << std::endl;


  cout << "calling pose only bundle adjustment by ceres pose only" << endl;
  Sophus::SE3d pose_only_ceres;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentPoseOnlyCeres(my_pts_3d_eigen, my_pts_2d_eigen , K_eigen, pose_only_ceres);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by ceres cost time: " << time_used.count() << " seconds." << endl;


  cout << "calling bundle adjustment by ceres" << endl;
  Sophus::SE3d pose_ceres(pose_only_ceres);
  t1 = chrono::steady_clock::now();
  bundleAdjustmentCeres(my_pts_3d_eigen, my_pts_2d_eigen , K_eigen, pose_ceres);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve BA by ceres cost time: " << time_used.count() << " seconds." << endl;


  cout << "calling pose only bundle adjustment by ceres auto diff" << endl;
  Sophus::SE3d pose_only_ceres_auto_diff;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentPoseOnlyCeresAutoDiff(my_pts_3d_eigen, my_pts_2d_eigen , K_eigen, pose_only_ceres_auto_diff);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by ceres auto diff cost time: " << time_used.count() << " seconds." << endl;


  cout << "calling bundle adjustment by ceres auto diff" << endl;
  Sophus::SE3d pose_ceres_auto_diff(pose_only_ceres_auto_diff);
  t1 = chrono::steady_clock::now();
  bundleAdjustmentCeresAutoDiff(my_pts_3d_eigen, my_pts_2d_eigen , K_eigen, pose_ceres_auto_diff);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by ceres auto diff cost time: " << time_used.count() << " seconds." << endl;



  std::cout << "only pose estimation: \n" << pose_only_ceres.matrix() << std::endl;



  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

      Eigen::Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();
      Eigen::Matrix<double, 2, 6> J;
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
  }

  cout << "pose by g-n: \n" << pose.matrix() << endl;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};


class reprojectionCostFunctionForPoseAndWorldPoint : public ceres::SizedCostFunction<
2,  /* number of residuals, which 2D observations */
6,  /* number of parameters for first parameter block, which is 6D pose parameters */
3   /* number of parameters for second parameter block, which is 3D world point parameters*/
> {


public:

  reprojectionCostFunctionForPoseAndWorldPoint(const Eigen::Vector2d &obs, const Eigen::Matrix<double, 3, 3> &K ) : obs_(obs), K_(K){}

  virtual bool Evaluate(double const * const * parameters,
                        double* residuals,
                        double** jacobians) const {
      // se3 for pose
      // Eigen::Matrix<double, 6, 1> se3_pose;
      // se3_pose << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5];

      Eigen::Map<const Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> se3_pose(parameters[0]);
      // Eigen::Matrix<double, 1, 6> se3_pose(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
      // se3_pose << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5];
      Eigen::Map<const Eigen::Vector3d> p_w(parameters[1]);

      Sophus::SE3d T = Sophus::SE3d::exp(se3_pose);

      const Eigen::Vector3d p_c = T * p_w;
      Eigen::Vector3d reproject_pixel = K_* p_c;

      const double X = p_c[0];
      const double Y = p_c[1];
      const double Z = p_c[2];
      const double inv_Z = 1.0 / ( p_c[2] + 1e-18 );
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
          // std::cout << "just test\n" << jacobian_res_by_pose << std::endl;
        }
        if(jacobians[1]){
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_res_by_p_w(jacobians[1]);
          Eigen::Matrix<double, 2, 3> jacobian_res_by_p_c;
          jacobian_res_by_p_c << J00, 0  , J02,
                                 0  , J11, J12;
          jacobian_res_by_p_w = jacobian_res_by_p_c * T.rotationMatrix();
        }


      }



      return true;

  }

private:
  const Eigen::Vector2d obs_;
  const Eigen::Matrix<double, 3, 3> K_;

};


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

      Sophus::SE3d T = Sophus::SE3d::exp(se3_pose);

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

struct reprojectionAutoDiffCostFunctor {


  reprojectionAutoDiffCostFunctor(const Eigen::Vector2d &obs, const Eigen::Matrix<double, 3, 3> &K ) : obs_(obs), K_(K){}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const p_w,
                  T* residuals) const {

      T p_c[3];

      ceres::AngleAxisRotatePoint(camera, p_w, p_c);
      p_c[0] += camera[3];
      p_c[1] += camera[4];
      p_c[2] += camera[5];

      T fx = T(K_(0,0));
      T fy = T(K_(1,1));
      T cx = T(K_(0,2));
      T cy = T(K_(1,2));

      T x = p_c[0] / p_c[2];
      T y = p_c[1] / p_c[2];

      T reproject_u = fx * x + cx;
      T reproject_v = fy * y + cy;


      residuals[0] = T(obs_[0]) - reproject_u;
      residuals[1] = T(obs_[1]) - reproject_v;



      return true;

  }

private:
  const Eigen::Vector2d obs_;
  const Eigen::Matrix<double, 3, 3> K_;

};


struct reprojectionAutoDiffCostFunctorForPoseOnly {


  reprojectionAutoDiffCostFunctorForPoseOnly(const Eigen::Vector3d &p_w, const Eigen::Vector2d &obs, const Eigen::Matrix<double, 3, 3> &K ) : obs_(obs), K_(K), p_w_(p_w){}

  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {

      T p_c[3];
      T p_w[3];
      p_w[0] = T(p_w_[0]);
      p_w[1] = T(p_w_[1]);
      p_w[2] = T(p_w_[2]);

      ceres::AngleAxisRotatePoint(camera, p_w, p_c);
      p_c[0] += camera[3];
      p_c[1] += camera[4];
      p_c[2] += camera[5];

      T fx = T(K_(0,0));
      T fy = T(K_(1,1));
      T cx = T(K_(0,2));
      T cy = T(K_(1,2));

      T x = p_c[0] / p_c[2];
      T y = p_c[1] / p_c[2];

      T reproject_u = fx * x + cx;
      T reproject_v = fy * y + cy;


      residuals[0] = T(obs_[0]) - reproject_u;
      residuals[1] = T(obs_[1]) - reproject_v;



      return true;

  }

private:
  const Eigen::Vector3d p_w_;
  const Eigen::Vector2d obs_;
  const Eigen::Matrix<double, 3, 3> K_;

};



void bundleAdjustmentCeres(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
) {
  

  std::vector<Eigen::Vector3d> points_3d_copy(points_3d);

  Eigen::Matrix<double, 1, 6> se3_vec = pose.log().transpose();

  ceres::Problem problem;
  
  for(int idx=0;idx<points_2d.size();idx++){


    ceres::CostFunction *cost_function = new reprojectionCostFunctionForPoseAndWorldPoint(points_2d[idx], K);
    problem.AddResidualBlock(cost_function, nullptr, se3_vec.data(), points_3d[idx].data());
  }
  
  

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  // options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  // se3_vec << pose_array[0], pose_array[1], pose_array[2], pose_array[3], pose_array[4], pose_array[5]; 

  Sophus::SE3d estimated_pose = Sophus::SE3d::exp(se3_vec);

  // for(int i=0;i<points_3d_copy.size();i++){


  //   std::cout << "3d point before:" << points_3d_copy[i].x() << " , " <<  points_3d_copy[i].y() << " , " << points_3d_copy[i].z() << "| "
  //   << "after:" << points_3d[i].x() << " , " <<  points_3d[i].y() << " , " << points_3d[i].z()
  //   << std::endl;

  // }

  std::cout << "pose estimation: \n" << estimated_pose.matrix() << std::endl;


}


void bundleAdjustmentCeresAutoDiff(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
) {
  
  Eigen::Matrix3d rotation_matrix = pose.rotationMatrix();
  Eigen::Vector3d translation = pose.translation();
  Eigen::AngleAxisd angle_axis(rotation_matrix);
  Eigen::Vector3d rotation_vec = angle_axis.angle() * angle_axis.axis();

  std::vector<Eigen::Vector3d> points_3d_copy(points_3d);


  double pose_array[6];
  for (int i = 0; i < 3; ++i) {
      pose_array[i] = rotation_vec[i];
      pose_array[i + 3] = translation[i];
  }
 

  // Eigen::Matrix<double, 1, 6> se3_vec = pose.log().transpose();

  ceres::Problem problem;
  
  for(int idx=0;idx<points_2d.size();idx++){


    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<reprojectionAutoDiffCostFunctor, 2, 6, 3>(
      new reprojectionAutoDiffCostFunctor(points_2d[idx], K)
    );
    problem.AddResidualBlock(cost_function, nullptr, pose_array, points_3d[idx].data());
  }

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  // options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";


  rotation_vec = Eigen::Vector3d(pose_array[0], pose_array[1], pose_array[2]);
  translation = Eigen::Vector3d(pose_array[3], pose_array[4], pose_array[5]);

  Eigen::Vector3d axis = rotation_vec.normalized();
  double angle = rotation_vec.norm();
  angle_axis = Eigen::AngleAxisd(angle, axis);
  rotation_matrix = angle_axis.toRotationMatrix();

  pose = Sophus::SE3d(rotation_matrix, translation);


  for(int i=0;i<points_3d_copy.size();i++){


    std::cout << "3d point before:" << points_3d_copy[i].x() << " , " <<  points_3d_copy[i].y() << " , " << points_3d_copy[i].z() << "| "
    << "after:" << points_3d[i].x() << " , " <<  points_3d[i].y() << " , " << points_3d[i].z()
    << std::endl;

  }


  std::cout << "BA estimation with auto diff: \n" << pose.matrix() << std::endl;


}

void bundleAdjustmentPoseOnlyCeres(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
) {
  
  Eigen::Matrix<double, 1, 6> se3_vec = pose.log().transpose();

  ceres::Problem problem;
  
  for(int idx=0;idx<points_2d.size();idx++){


    ceres::CostFunction *cost_function = new reprojectionCostFunctionForPoseOnly(points_3d[idx], points_2d[idx], K);
    problem.AddResidualBlock(cost_function, nullptr, se3_vec.data());
  }
  
  

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  // options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";


  pose = Sophus::SE3d::exp(se3_vec);
  std::cout << "only pose estimation: \n" << pose.matrix() << std::endl;


}


void bundleAdjustmentPoseOnlyCeresAutoDiff(
  std::vector<Eigen::Vector3d> &points_3d,
  std::vector<Eigen::Vector2d> &points_2d,
  const Eigen::Matrix<double, 3, 3> &K,
  Sophus::SE3d &pose
) {
  
  Eigen::Matrix3d rotation_matrix = pose.rotationMatrix();
  Eigen::Vector3d translation = pose.translation();
  Eigen::AngleAxisd angle_axis(rotation_matrix);
  Eigen::Vector3d rotation_vec = angle_axis.angle() * angle_axis.axis();

  double pose_array[6];
  for (int i = 0; i < 3; ++i) {
      pose_array[i] = rotation_vec[i];
      pose_array[i + 3] = translation[i];
  }

  // Eigen::Matrix<double, 1, 6> se3_vec = pose.log().transpose();

  ceres::Problem problem;
  
  for(int idx=0;idx<points_2d.size();idx++){


    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<reprojectionAutoDiffCostFunctorForPoseOnly, 2, 6>(
      new reprojectionAutoDiffCostFunctorForPoseOnly(points_3d[idx], points_2d[idx], K)
    );
    problem.AddResidualBlock(cost_function, nullptr, pose_array);
  }

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  // options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";


  rotation_vec = Eigen::Vector3d(pose_array[0], pose_array[1], pose_array[2]);
  translation = Eigen::Vector3d(pose_array[3], pose_array[4], pose_array[5]);

  Eigen::Vector3d axis = rotation_vec.normalized();
  double angle = rotation_vec.norm();
  angle_axis = Eigen::AngleAxisd(angle, axis);
  rotation_matrix = angle_axis.toRotationMatrix();

  pose = Sophus::SE3d(rotation_matrix, translation);

  std::cout << "only pose estimation with auto diff: \n" << pose.matrix() << std::endl;


}

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}
