project(imageBasics)

add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})


add_executable(undistortEquidistantImage undistortEquidistantImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortEquidistantImage ${OpenCV_LIBS})


add_executable(undistortEquidistantImageWithOptimalCrop undistortEquidistantImageWithOptimalCrop.cpp)
# 链接OpenCV库
target_link_libraries(undistortEquidistantImageWithOptimalCrop ${OpenCV_LIBS})

add_executable(undistortImageWithOptimalCoarseCrop undistortImageWithOptimalCoarseCrop.cpp)
# 链接OpenCV库
target_link_libraries(undistortImageWithOptimalCoarseCrop ${OpenCV_LIBS})

add_executable(undistortImageWithOptimalCrop undistortImageWithOptimalCrop.cpp)
# 链接OpenCV库
target_link_libraries(undistortImageWithOptimalCrop ${OpenCV_LIBS})