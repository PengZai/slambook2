#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确

int main(int argc, char **argv) {

  // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
  // 畸变参数
    float k1 = -0.04345139283609733, k2 = 0.019439878275353862, k3 = -0.03544505860721041, k4 = 0.022121647569599227;
    // 内参
    float fx = 1059.6087870662222, fy = 1059.5973705610731, cx = 1050.3552896004385, cy = 731.3912426749002;

  cv::Mat image = cv::imread(argv[1], 0);   // 图像是灰度图，CV_8UC1
  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

  // 计算去畸变后图像的内容
  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
      double x = (u - cx) / fx;
      double y = (v - cy) / fy;

      double r_u = std::sqrt(x * x + y * y);
      if (r_u < 1e-8) {
        // center pixel
        image_undistort.at<uchar>(v, u) = image.at<uchar>(cv::Point(u, v));
        continue;
      }

      // Compute theta (angle from optical axis)
      double theta = std::atan(r_u);

      // Distorted angle
      double theta_d = theta * (1 + k1 * pow(theta, 2) + k2 * pow(theta, 4) +
                                k3 * pow(theta, 6) + k4 * pow(theta, 8));

      // Mapping back to distorted normalized coordinates
      double scale = (theta_d / r_u);
      double x_d = x * scale;
      double y_d = y * scale;

      // Convert to pixel coordinates
      double u_distorted = fx * x_d + cx;
      double v_distorted = fy * y_d + cy;

      // 赋值 (最近邻插值)
      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
        image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
      } else {
        image_undistort.at<uchar>(v, u) = 0;
      }
    }
  }

  // 画图去畸变后图像
  cv::imshow("distorted", image);
  cv::imshow("undistorted", image_undistort);
  cv::imwrite("undistorted.png", image_undistort);
  cv::waitKey();
  return 0;
}
