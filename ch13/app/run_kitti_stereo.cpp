//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

// DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv) {
    // google::ParseCommandLineFlags(&argc, &argv, true);

    std::string config_file_path = argv[1];
    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(config_file_path));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
