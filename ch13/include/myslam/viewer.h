//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

/**
 * 可视化
 */
class Viewer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    void SetMap(Map::Ptr map) { map_ = map; }

    void Close();

    // 增加一个当前帧
    void AddCurrentFrame(Frame::Ptr current_frame);

    // 更新地图
    void UpdateMap();

   private:
    void ThreadLoop();

    void DrawFrame(SE3 Twc, const float* color);

    void DrawMapPoints();
    void drawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3i &bgr, const int line_size);
    void drawPoint(const Eigen::Vector3d &pt3d, const Eigen::Vector3i &bgr, const int point_size);

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
    bool map_updated_ = false;
    std::vector<Vec3> keyframe_trajs_;
    std::vector<Vec3> GT_trajs_;

    std::mutex viewer_data_mutex_;
};
}  // namespace myslam

#endif  // MYSLAM_VIEWER_H
