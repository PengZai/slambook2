#include "myslam/dataset.h"
#include "myslam/frame.h"
#include <dirent.h>
#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    // read camera intrinsics and extrinsics
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

  
    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * t;
        K = K * 0.5;
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    current_image_index_ = 0;
    return true;
}


bool Dataset::isImageFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "bmp";
}



bool Dataset::readImageFromDir(const std::string dir_path, std::vector<std::string> &images_names){


    DIR* dir = opendir(dir_path.c_str());
    struct dirent* entry;

    if (dir == nullptr) {
        std::cerr << "Cannot open directory: " << dir_path << std::endl;
        return false;
    }

    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (entry->d_type == DT_REG && isImageFile(name)) {
            images_names.push_back(name);
        }
    }

    return true;

}

bool Dataset::Init_for_Botanic_Garden() {


    std::string left_image_dir = dataset_path_ + "/left_rgb_rectified";
    std::string right_image_dir = dataset_path_ + "/right_rgb_rectified";

    readImageFromDir(left_image_dir, left_image_names_);
    readImageFromDir(right_image_dir, right_image_names_);


    Mat33 K;
    K << 654.78839288, 0.0, 462.13834,
         0.0, 654.78839288, 306.05381012,
         0.0, 0.0, 1.0;
    
    Vec3 t1;
    t1 << 0, 0, 0;
    Camera::Ptr new_camera1(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t1.norm(), SE3(SO3(), t1)));

    Vec3 t2;
    t2 << -0.253736175410149, 0, 0;
    Camera::Ptr new_camera2(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t2.norm(), SE3(SO3(), t2)));


    cameras_.push_back(new_camera1);
    cameras_.push_back(new_camera2);

    current_image_index_ = 0;

    return true;


}

Frame::Ptr Dataset::NextFrame() {
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;
    // read images
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }

    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;
    current_image_index_++;
    return new_frame;
}


Frame::Ptr Dataset::NextFrameForBotanicGarden(){


    cv::Mat image_left, image_right;
    // read images
    image_left =
        cv::imread(dataset_path_ + "/left_rgb_rectified/" + left_image_names_[current_image_index_],
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread(dataset_path_ + "/right_rgb_rectified/" + right_image_names_[current_image_index_],
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }


    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left;
    new_frame->right_img_ = image_right;
    current_image_index_++;
    return new_frame;

}

}  // namespace myslam