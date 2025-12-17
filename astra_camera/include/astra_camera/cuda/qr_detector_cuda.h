#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include "cuda_utils.h"

namespace astra_camera {
namespace cuda {

struct QRCodeResult {
    bool detected;
    std::string data;
    std::vector<cv::Point2f> corners;
    cv::Mat rvec;
    cv::Mat tvec;
    float distance;
    float angle;
};

class QRDetectorCUDA {
public:
    QRDetectorCUDA(float qr_size = 0.1f);
    ~QRDetectorCUDA();
    
    bool detectAndDecode(
        const cv::Mat& rgb_image,
        const cv::Mat& depth_image,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        QRCodeResult& result);
    
    void setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    void setQRSize(float size) { qr_size_ = size; }
    
private:
    float qr_size_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    cv::QRCodeDetector qr_detector_;
    
    bool estimatePoseWithDepth(
        const std::vector<cv::Point2f>& corners,
        const cv::Mat& depth_image,
        cv::Mat& rvec,
        cv::Mat& tvec,
        float& distance,
        float& angle);
    
    float getDepthAtPoint(const cv::Mat& depth_image, const cv::Point2f& point);
};

}
}
