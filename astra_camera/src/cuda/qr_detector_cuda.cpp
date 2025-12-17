#include "astra_camera/cuda/qr_detector_cuda.h"
#include <opencv2/calib3d.hpp>

namespace astra_camera {
namespace cuda {

QRDetectorCUDA::QRDetectorCUDA(float qr_size)
    : qr_size_(qr_size)
{
}

QRDetectorCUDA::~QRDetectorCUDA() {
}

void QRDetectorCUDA::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
}

float QRDetectorCUDA::getDepthAtPoint(const cv::Mat& depth_image, const cv::Point2f& point) {
    int x = static_cast<int>(point.x + 0.5f);
    int y = static_cast<int>(point.y + 0.5f);
    
    if (x < 0 || x >= depth_image.cols || y < 0 || y >= depth_image.rows) {
        return 0.0f;
    }
    
    int radius = 3;
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < depth_image.cols && ny >= 0 && ny < depth_image.rows) {
                uint16_t depth = depth_image.at<uint16_t>(ny, nx);
                if (depth > 0 && depth < 10000) {
                    sum += depth * 0.001f;
                    count++;
                }
            }
        }
    }
    
    return (count > 0) ? (sum / count) : 0.0f;
}

bool QRDetectorCUDA::estimatePoseWithDepth(
    const std::vector<cv::Point2f>& corners,
    const cv::Mat& depth_image,
    cv::Mat& rvec,
    cv::Mat& tvec,
    float& distance,
    float& angle)
{
    if (corners.size() != 4) return false;
    
    std::vector<cv::Point3f> object_points;
    object_points.push_back(cv::Point3f(0, 0, 0));
    object_points.push_back(cv::Point3f(qr_size_, 0, 0));
    object_points.push_back(cv::Point3f(qr_size_, qr_size_, 0));
    object_points.push_back(cv::Point3f(0, qr_size_, 0));
    
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center.x += corner.x;
        center.y += corner.y;
    }
    center.x /= 4.0f;
    center.y /= 4.0f;
    
    distance = getDepthAtPoint(depth_image, center);
    
    if (distance < 0.1f || distance > 5.0f) {
        return false;
    }
    
    bool success = cv::solvePnP(
        object_points,
        corners,
        camera_matrix_,
        dist_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE);
    
    if (success) {
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        angle = atan2(rotation_matrix.at<double>(1, 0), 
                     rotation_matrix.at<double>(0, 0)) * 180.0 / CV_PI;
    }
    
    return success;
}

bool QRDetectorCUDA::detectAndDecode(
    const cv::Mat& rgb_image,
    const cv::Mat& depth_image,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    QRCodeResult& result)
{
    result.detected = false;
    
    if (rgb_image.empty()) return false;
    
    setCameraParams(camera_matrix, dist_coeffs);
    
    std::vector<cv::Point> points;
    std::string decoded_info = qr_detector_.detectAndDecode(rgb_image, points);
    
    if (decoded_info.empty() || points.empty()) {
        return false;
    }
    
    result.detected = true;
    result.data = decoded_info;
    
    for (const auto& pt : points) {
        result.corners.push_back(cv::Point2f(pt.x, pt.y));
    }
    
    if (!depth_image.empty() && result.corners.size() == 4) {
        estimatePoseWithDepth(
            result.corners,
            depth_image,
            result.rvec,
            result.tvec,
            result.distance,
            result.angle);
    }
    
    return true;
}

}
}
